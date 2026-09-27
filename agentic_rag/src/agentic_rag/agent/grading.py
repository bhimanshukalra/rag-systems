from pydantic import BaseModel, Field

from agentic_rag.generation.synthesizer import RETRY_ATTEMPTS, get_llm

GRADING_PROMPT = (
    "You judge whether retrieved evidence is sufficient to answer a "
    "question. Given a question and the evidence retrieved so far, decide "
    "whether the evidence contains enough information to answer "
    "accurately. If it doesn't, state clearly what's missing."
)


class EvidenceGrade(BaseModel):
    sufficient: bool = Field(
        description="True if the evidence is enough to answer the question accurately."
    )
    reason: str = Field(
        description="Brief explanation of the judgment, including what's missing if insufficient."
    )


def grade_evidence(question: str, evidence_text: str, *, model: str) -> EvidenceGrade:
    # with_structured_output() must come before with_retry() -- the retry
    # wrapper drops chat-model-specific methods, so calling it first would
    # make with_structured_output unavailable (same reasoning documented
    # on synthesizer.py's with_retry call site).
    llm = get_llm(model).with_structured_output(EvidenceGrade).with_retry(
        stop_after_attempt=RETRY_ATTEMPTS
    )

    return llm.invoke(
        [
            ("system", GRADING_PROMPT),
            ("human", f"Question: {question}\n\nEvidence:\n{evidence_text}"),
        ]
    )
