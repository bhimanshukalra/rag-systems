from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlmodel import Session

from agentic_rag.agent.grading import grade_evidence
from agentic_rag.config import Settings
from agentic_rag.generation.synthesizer import RETRY_ATTEMPTS, generate_answer, get_llm
from agentic_rag.retrieval.pipeline import retrieve
from agentic_rag.retrieval.web_search import search_web

REWRITE_PROMPT = (
    "You reformulate a search query that failed to retrieve sufficient "
    "evidence. Given the original query and why the evidence was "
    "insufficient, write ONE improved search query. Return only the "
    "query text, nothing else."
)


class RetrieveInput(BaseModel):
    query: str = Field(description="The search query to retrieve passages for.")


class GradeEvidenceInput(BaseModel):
    question: str = Field(description="The original user question.")
    evidence: str = Field(description="The retrieved evidence text to judge.")


class WebSearchInput(BaseModel):
    query: str = Field(description="The web search query.")


class RewriteQueryInput(BaseModel):
    original_query: str = Field(
        description="The query that returned insufficient evidence."
    )
    reason: str = Field(description="Why the evidence was insufficient.")


class GenerateAnswerInput(BaseModel):
    question: str = Field(description="The original user question.")
    evidence: str = Field(
        description=(
            "The gathered evidence to base the final answer on -- compile "
            "the relevant excerpts from earlier tool results."
        )
    )


def _format_documents(documents: list[Document]) -> str:
    if not documents:
        return "No results found."
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def build_tools(
    *, session: Session, vector_store, settings: Settings
) -> list[StructuredTool]:
    """Build the agent's five tools, each closing over the app-level
    dependencies (session, vector_store, settings) it needs -- the LLM
    only ever supplies the arguments declared in each tool's own schema.
    """

    def _hybrid_retrieve(query: str) -> str:
        documents = retrieve(
            query, session=session, vector_store=vector_store, settings=settings
        )
        return _format_documents(documents)

    def _grade_evidence(question: str, evidence: str) -> str:
        grade = grade_evidence(question, evidence, model=settings.llm_model)
        return f"sufficient={grade.sufficient}; reason={grade.reason}"

    def _web_search(query: str) -> str:
        return search_web(query, api_key=settings.tavily_api_key)

    def _rewrite_query(original_query: str, reason: str) -> str:
        llm = get_llm(settings.llm_model).with_retry(stop_after_attempt=RETRY_ATTEMPTS)
        response = llm.invoke(
            [
                ("system", REWRITE_PROMPT),
                (
                    "human",
                    f"Original query: {original_query}\nWhy insufficient: {reason}",
                ),
            ]
        )
        return response.content.strip()

    def _generate_answer(question: str, evidence: str) -> str:
        # Evidence here is text the agent compiled from earlier tool
        # results, not the original Document objects -- wrapped as one
        # synthetic Document so this can reuse generate_answer's citation
        # formatting. Trade-off: the "Sources:" list shows a generic
        # label rather than real per-chunk sources -- not worth solving
        # until the live smoke test (step 8) shows it actually matters.
        document = Document(
            page_content=evidence, metadata={"source": "agent-gathered evidence"}
        )
        result = generate_answer(question, [document], model=settings.llm_model)
        return result.answer

    return [
        StructuredTool.from_function(
            func=_hybrid_retrieve,
            name="hybrid_retrieve",
            description=(
                "Retrieve relevant passages from the knowledge base for a "
                "search query. Use this first for any question."
            ),
            args_schema=RetrieveInput,
        ),
        StructuredTool.from_function(
            func=_grade_evidence,
            name="grade_evidence",
            description=(
                "Judge whether gathered evidence is sufficient to answer "
                "the question accurately. Use this after retrieving."
            ),
            args_schema=GradeEvidenceInput,
        ),
        StructuredTool.from_function(
            func=_web_search,
            name="web_search",
            description=(
                "Search the web. Use this when grade_evidence says the "
                "knowledge base's evidence is insufficient."
            ),
            args_schema=WebSearchInput,
        ),
        StructuredTool.from_function(
            func=_rewrite_query,
            name="rewrite_query",
            description=(
                "Reformulate a query that returned insufficient evidence, "
                "as an alternative to falling back to web_search."
            ),
            args_schema=RewriteQueryInput,
        ),
        StructuredTool.from_function(
            func=_generate_answer,
            name="generate_answer",
            description=(
                "Produce the final answer from gathered evidence. Calling "
                "this tool ends your turn -- only call it once you have "
                "sufficient evidence."
            ),
            args_schema=GenerateAnswerInput,
        ),
    ]
