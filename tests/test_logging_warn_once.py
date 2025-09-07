from __future__ import annotations


def test_log_warn_once_prints_once(capsys) -> None:
    # Arrange: reset seen tags to isolate this test
    from timewarp.utils import logging as twlog

    try:
        twlog._seen_tags.clear()
    except Exception:  # pragma: no cover - defensive
        pass

    # Act: emit the same tag multiple times
    twlog.log_warn_once("unit.test.tag", Exception("boom"), {"a": 1})
    twlog.log_warn_once("unit.test.tag", Exception("second"), {"a": 2})
    twlog.log_warn_once("unit.test.tag")

    # Assert: printed exactly once
    out = capsys.readouterr().out
    assert out.count("[timewarp][warn] unit.test.tag") == 1


def test_log_warn_once_different_tags(capsys) -> None:
    from timewarp.utils import logging as twlog

    try:
        twlog._seen_tags.clear()
    except Exception:  # pragma: no cover - defensive
        pass

    # Emit two different tags
    twlog.log_warn_once("unit.test.tagA", Exception("a"))
    twlog.log_warn_once("unit.test.tagB", Exception("b"))

    out = capsys.readouterr().out
    assert out.count("[timewarp][warn] unit.test.tagA") == 1
    assert out.count("[timewarp][warn] unit.test.tagB") == 1
