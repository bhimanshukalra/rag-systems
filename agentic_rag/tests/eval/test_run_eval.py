from langchain_core.documents import Document
from run_eval import (
    RECALL_KS,
    filter_documents,
    filter_questions,
    ranked_distinct_sources,
    score_question,
)


def test_filter_questions_drops_any_question_touching_an_excluded_source():
    questions = [
        {"question": "a", "expected_sources": ["s1"]},
        {"question": "b", "expected_sources": ["s2"]},
        {"question": "multi", "expected_sources": ["s1", "s2"]},
        {"question": "none", "expected_sources": []},
    ]

    kept = filter_questions(questions, frozenset({"s2"}))

    assert [q["question"] for q in kept] == ["a", "none"]


def test_filter_documents_removes_chunks_from_excluded_sources():
    docs = [
        Document(page_content="x", metadata={"source": "s1"}),
        Document(page_content="y", metadata={"source": "s2"}),
    ]

    kept = filter_documents(docs, frozenset({"s2"}))

    assert [d.metadata["source"] for d in kept] == ["s1"]


def test_filters_are_noops_with_nothing_excluded():
    questions = [{"question": "a", "expected_sources": ["s1"]}]
    docs = [Document(page_content="x", metadata={"source": "s1"})]

    assert filter_questions(questions, frozenset()) == questions
    assert filter_documents(docs, frozenset()) == docs


def test_ranked_distinct_sources_dedupes_preserving_first_occurrence_order():
    docs = [
        Document(page_content="a", metadata={"source": "s1"}),
        Document(page_content="b", metadata={"source": "s2"}),
        Document(page_content="c", metadata={"source": "s1"}),
        Document(page_content="d", metadata={"source": "s3"}),
    ]

    assert ranked_distinct_sources(docs) == ["s1", "s2", "s3"]


def test_score_question_hit_at_rank_one():
    recalls, rr, hit_rank = score_question(["s1", "s2", "s3"], ["s1"])

    assert hit_rank == 1
    assert rr == 1.0
    assert recalls == {k: 1 for k in RECALL_KS}


def test_score_question_hit_at_rank_four():
    recalls, rr, hit_rank = score_question(["s2", "s3", "s4", "s1"], ["s1"])

    assert hit_rank == 4
    assert rr == 0.25
    assert recalls[1] == 0
    assert recalls[4] == 1
    assert recalls[10] == 1


def test_score_question_no_hit_within_pool():
    recalls, rr, hit_rank = score_question(["s2", "s3"], ["s1"])

    assert hit_rank is None
    assert rr == 0.0
    assert all(v == 0 for v in recalls.values())


def test_score_question_matches_any_of_multiple_expected_sources():
    _recalls, rr, hit_rank = score_question(["s2", "s5"], ["s1", "s5"])

    assert hit_rank == 2
    assert rr == 0.5
