from agentic_rag.api.rate_limit import FixedWindowLimiter


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_allows_up_to_limit_then_blocks():
    limiter = FixedWindowLimiter(3, 60, _Clock())

    assert [limiter.acquire() for _ in range(3)] == [None, None, None]
    assert limiter.acquire() == 60


def test_retry_after_shrinks_as_window_elapses():
    clock = _Clock()
    limiter = FixedWindowLimiter(1, 60, clock)
    limiter.acquire()
    clock.now = 45

    assert limiter.acquire() == 15


def test_window_resets_after_expiry():
    clock = _Clock()
    limiter = FixedWindowLimiter(1, 60, clock)
    limiter.acquire()
    assert limiter.acquire() is not None

    clock.now = 60

    assert limiter.acquire() is None
