from claude_recall.parsers.claude_code import extract_text


def test_extract_text_string():
    assert extract_text("hello") == ("hello", ())


def test_extract_text_blocks():
    content = [
        {"type": "text", "text": "hi"},
        {"type": "tool_use", "name": "Bash"},
        {"type": "tool_result", "content": [{"type": "text", "text": "ok"}]},
    ]
    text, tools = extract_text(content)
    assert "hi" in text
    assert "[tool_use:Bash]" in text
    assert "[tool_result] ok" in text
    assert tools == ("Bash",)


def test_extract_text_garbage():
    assert extract_text(None) == ("", ())
    assert extract_text(123) == ("", ())
