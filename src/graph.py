from typing import Literal

from langgraph.graph import END, START, StateGraph

from llm import get_llm, get_tavily_search_tool
from state import AgentState, EvidenceGrade, RouteDecision

# Nodes
def route_question(state: AgentState):
    question = state["question"]

    router_llm = get_llm().with_structured_output(RouteDecision, method="json_mode")

    decision = router_llm.invoke(f"""
You are a router for an Agentic RAG assistant.

Route to "kb" if the user asks about:
- Agentic RAG
- LangGraph Agentic RAG workflow
- retrieval grading
- query rewriting
- RAG architecture
- retriever tools
- web fallback in RAG

Route to "direct" only for greetings, thanks, or very simple conversation.

Question:
{question}

Return your response as valid JSON.
Example:
{{"route": "kb"}}
""")

    print("[Router]", decision.route)

    return {
        "current_query": question,
        "source_used": decision.route,
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    return "direct_answer"


def grade_kb_evidence(state: AgentState):
    question = state["question"]

    kb_grader_llm = get_llm().with_structured_output(EvidenceGrade, method="json_mode")

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
        for doc in state["kb_docs"]
    )

    grade = kb_grader_llm.invoke(f"""
You are an evidence grader.

Question:
{question}

Private KB evidence:
{context}

Can this private KB evidence answer the question?
Return "good" if it can answer.
Return "weak" if it cannot answer or is incomplete.

Return your response as valid JSON.
Example:
{{"grade": "good"}}
""")

    print("[KB Grader]", grade.grade)

    return {"kb_grade": grade.grade}


def decide_after_kb_grade(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    return "search_web"


def search_web(state: AgentState):
    query = state["current_query"]

    print(f"[Tavily Search] Query: {query}")

    result = get_tavily_search_tool().invoke({"query": query})

    # Tavily can return dict/list structures depending on package version.
    # Convert it into readable text for grading and generation.
    if isinstance(result, dict):
        answer = result.get("answer", "")
        results = result.get("results", [])
        lines = []
        if answer:
            lines.append(f"Tavily answer: {answer}")

        for item in results:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")

        web_text = "\n\n".join(lines) if lines else str(result)
    else:
        web_text = str(result)

    print("[Tavily Search] Result characters:", len(web_text))

    return {
        "web_results": web_text,
        "source_used": "web",
    }


def grade_web_evidence(state: AgentState):
    question = state["question"]
    web_results = state["web_results"]

    web_grader_llm = get_llm().with_structured_output(EvidenceGrade, method="json_mode")

    grade = web_grader_llm.invoke(f"""
You are an evidence grader.

Question:
{question}

Web search evidence:
{web_results}

Can this web evidence answer the question?
Return "good" if it can answer.
Return "weak" if it cannot answer or is incomplete.

Return your response as valid JSON.
Example:
{{"grade": "good"}}
""")

    print("[Web Grader]", grade.grade)

    return {"web_grade": grade.grade}


MAX_RETRIES = 1


def decide_after_web_grade(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"

    return "answer_insufficient"


def rewrite_query(state: AgentState):
    question = state["question"]
    retry_count = state["retry_count"] + 1

    rewritten = get_llm().invoke(f"""
Rewrite the question for better retrieval and web search.

Rules:
- Preserve original intent.
- Make it specific and search-friendly.
- Do not answer.
- Return only the rewritten query.

Original question:
{question}
""").content.strip()

    print("[Rewriter]", rewritten)

    return {
        "current_query": rewritten,
        "retry_count": retry_count,
    }


def generate_from_kb(state: AgentState):
    question = state["question"]

    context = "\n\n".join(
        f"[KB Source: {doc.metadata.get('source')}]\n{doc.page_content}"
        for doc in state["kb_docs"]
    )

    answer = get_llm().invoke(f"""
You are a technical instructor.

Answer using ONLY the private KB context.

Rules:
- Beginner-friendly explanation.
- Do not invent unsupported details.
- Mention that the answer is based on the private KB.
- Include source type: Private KB.

Question:
{question}

Private KB context:
{context}
""").content

    return {
        "answer": answer,
        "source_used": "private_kb",
    }


def generate_from_web(state: AgentState):
    question = state["question"]
    web_context = state["web_results"]

    answer = get_llm().invoke(f"""
You are a technical instructor.

The private KB was insufficient, so web search was used.

Answer using ONLY the web search context.

Rules:
- Beginner-friendly explanation.
- Do not invent unsupported details.
- Mention that the answer is based on Tavily web search.
- Include source type: Web Search.
- If URLs are present in context, include the most useful URLs.

Question:
{question}

Web search context:
{web_context}
""").content

    return {
        "answer": answer,
        "source_used": "web_search",
    }


def direct_answer(state: AgentState):
    question = state["question"]

    answer = get_llm().invoke(f"""
Respond briefly and naturally.

Message:
{question}
""").content

    return {
        "answer": answer,
        "source_used": "direct",
    }


def answer_insufficient(state: AgentState):
    answer = (
        "I could not find enough reliable evidence in the private knowledge base "
        "or the web search results to answer this confidently. "
        "Please provide more specific documents or rephrase the question."
    )

    return {
        "answer": answer,
        "source_used": "insufficient_evidence",
    }


def build_graph(retriever):
    def retrieve_kb(state: AgentState):
        query = state["current_query"]
        docs = retriever.invoke(query)

        print(f"[KB Retriever] Query: {query}")
        print(f"[KB Retriever] Retrieved: {len(docs)} chunks")

        return {"kb_docs": docs}

    workflow = StateGraph(AgentState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve_kb", retrieve_kb)
    workflow.add_node("grade_kb_evidence", grade_kb_evidence)
    workflow.add_node("search_web", search_web)
    workflow.add_node("grade_web_evidence", grade_web_evidence)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate_from_kb", generate_from_kb)
    workflow.add_node("generate_from_web", generate_from_web)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("answer_insufficient", answer_insufficient)

    workflow.add_edge(START, "route_question")

    workflow.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge("retrieve_kb", "grade_kb_evidence")

    workflow.add_conditional_edges(
        "grade_kb_evidence",
        decide_after_kb_grade,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    workflow.add_edge("search_web", "grade_web_evidence")

    workflow.add_conditional_edges(
        "grade_web_evidence",
        decide_after_web_grade,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "answer_insufficient": "answer_insufficient",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve_kb")
    workflow.add_edge("generate_from_kb", END)
    workflow.add_edge("generate_from_web", END)
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("answer_insufficient", END)

    app = workflow.compile()

    print("Industry-style Agentic RAG graph compiled.")

    return app


def ask_agent(app, question: str):
    initial_state: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 90)
    print("QUESTION:")
    print(question)
    print("\nSOURCE USED:")
    print(result["source_used"])
    print("\nFINAL ANSWER:")
    print(result["answer"])
    print("=" * 90)

    return result
