import logging

import groq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from sqlmodel import Session

from agentic_rag.agent.prompts import SYSTEM_PROMPT
from agentic_rag.agent.tools import build_tools
from agentic_rag.config import Settings
from agentic_rag.generation.synthesizer import RETRY_ATTEMPTS, get_llm
from agentic_rag.observability.tracing import get_tracer

logger = logging.getLogger(__name__)

FINISH_TOOL_NAME = "generate_answer"


def run_agent(question: str, *, session: Session, vector_store, settings: Settings) -> str:
    with get_tracer().start_as_current_span("agent.run") as span:
        span.set_attribute("question", question)
        return _run_agent(question, session, vector_store, settings)


def _run_agent(question: str, session: Session, vector_store, settings: Settings) -> str:
    """Run the ReAct tool-calling loop for one question, returning the
    final answer text.

    The agent decides whether, when, and how many times to call each
    tool -- there's no fixed edge from retrieve to grade to fallback.
    Terminates when it calls generate_answer (the "finish" signal), when
    it answers directly with no tool calls, or when agent_max_steps is
    exceeded (in which case a final answer is forced from whatever
    evidence was gathered, rather than a bare error).
    """
    tools = build_tools(session=session, vector_store=vector_store, settings=settings)
    tools_by_name = {tool.name: tool for tool in tools}

    llm = (
        get_llm(settings.llm_model)
        .bind_tools(tools)
        .with_retry(stop_after_attempt=RETRY_ATTEMPTS)
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for step in range(settings.agent_max_steps):
        try:
            with get_tracer().start_as_current_span("agent.step") as step_span:
                step_span.set_attribute("step", step)
                response = llm.invoke(messages)
        except groq.BadRequestError as exc:
            if "tool_use_failed" not in str(exc):
                raise
            # The model emitted a malformed tool call (seen live: it wrote
            # its answer text where the JSON arguments belong). That's a
            # deterministic 400, so retrying won't help -- answer from the
            # evidence gathered so far instead of crashing the request.
            logger.warning(
                "Model produced a malformed tool call; forcing a final answer: %s", exc
            )
            return _force_final_answer(question, messages, tools_by_name)
        messages.append(response)

        if not response.tool_calls:
            # The model answered directly without calling generate_answer
            # -- its own text is the final answer.
            return response.content

        for tool_call in response.tool_calls:
            result = _invoke_tool(tools_by_name, tool_call)
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            if tool_call["name"] == FINISH_TOOL_NAME:
                return str(result)

    logger.warning(
        "Agent exceeded agent_max_steps (%d) for question: %s",
        settings.agent_max_steps,
        question,
    )
    return _force_final_answer(question, messages, tools_by_name)


def _force_final_answer(question: str, messages: list, tools_by_name: dict) -> str:
    evidence = "\n\n".join(
        str(message.content) for message in messages if isinstance(message, ToolMessage)
    )
    return str(
        tools_by_name[FINISH_TOOL_NAME].invoke(
            {"question": question, "evidence": evidence or "No evidence was gathered."}
        )
    )


def _invoke_tool(tools_by_name: dict, tool_call: dict) -> str:
    tool = tools_by_name.get(tool_call["name"])
    if tool is None:
        return f"Unknown tool: {tool_call['name']}"

    try:
        with get_tracer().start_as_current_span(f"tool.{tool_call['name']}"):
            return tool.invoke(tool_call["args"])
    except Exception as exc:
        # A single tool failing shouldn't crash the whole turn -- feed the
        # error back so the agent can react (retry, rewrite, fall back).
        logger.exception("Tool %s failed", tool_call["name"])
        return f"Tool error: {exc}"
