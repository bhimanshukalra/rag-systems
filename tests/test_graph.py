from graph import (
    MAX_RETRIES,
    decide_after_kb_grade,
    decide_after_web_grade,
    route_after_router,
)


def test_route_after_router_routes_kb_to_retrieve_kb():
    assert route_after_router({"source_used": "kb"}) == "retrieve_kb"


def test_route_after_router_routes_anything_else_to_direct_answer():
    assert route_after_router({"source_used": "direct"}) == "direct_answer"


def test_decide_after_kb_grade_good_generates_from_kb():
    assert decide_after_kb_grade({"kb_grade": "good"}) == "generate_from_kb"


def test_decide_after_kb_grade_weak_searches_web():
    assert decide_after_kb_grade({"kb_grade": "weak"}) == "search_web"


def test_decide_after_web_grade_good_generates_from_web():
    state = {"web_grade": "good", "retry_count": 0}
    assert decide_after_web_grade(state) == "generate_from_web"


def test_decide_after_web_grade_weak_rewrites_query_while_retries_remain():
    state = {"web_grade": "weak", "retry_count": MAX_RETRIES - 1}
    assert decide_after_web_grade(state) == "rewrite_query"


def test_decide_after_web_grade_weak_gives_up_once_retries_exhausted():
    state = {"web_grade": "weak", "retry_count": MAX_RETRIES}
    assert decide_after_web_grade(state) == "answer_insufficient"
