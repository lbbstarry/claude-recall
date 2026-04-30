from datetime import UTC, datetime

from claude_recall.search.filters import parse


def test_parse_no_filters():
    q, f = parse("how to use sqlite-vec")
    assert q == "how to use sqlite-vec"
    assert f.project is None and f.since_iso is None and f.role is None and f.tool is None


def test_parse_known_filters():
    q, f = parse("auth bug project:myapp since:7d role:assistant tool:Bash")
    assert q == "auth bug"
    assert f.project == "myapp"
    assert f.role == "assistant"
    assert f.tool == "Bash"
    parsed = datetime.fromisoformat(f.since_iso)
    assert (datetime.now(UTC) - parsed).days <= 7


def test_parse_unknown_passes_through():
    q, f = parse("foo bar:baz hello project:x")
    assert "bar:baz" in q
    assert f.project == "x"


def test_parse_quoted_value():
    q, f = parse('hello project:"my app"')
    assert f.project == "my app"
    assert q == "hello"


def test_parse_invalid_since_returns_none():
    _, f = parse("hello since:nonsense")
    assert f.since_iso is None
