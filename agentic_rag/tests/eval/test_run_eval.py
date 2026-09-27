from langchain_core.documents import Document
from run_eval import RECALL_KS, ranked_distinct_sources, score_question


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
