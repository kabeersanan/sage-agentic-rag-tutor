from src.agents.quiz_agent import clean_json_text

def test_clean_json_with_markdown():
    dirty_json = "```json\n{\"question\": \"What is 2+2?\", \"answer\": \"4\"}\n```"
    result = clean_json_text(dirty_json)
    assert result == "{\"question\": \"What is 2+2?\", \"answer\": \"4\"}"

def test_clean_json_without_markdown():
    clean = "{\"question\": \"What is 2+2?\", \"answer\": \"4\"}"
    result = clean_json_text(clean)
    assert result == clean

def test_clean_json_with_plain_backticks():
    dirty_json = "```\n{\"key\": \"value\"}\n```"
    result = clean_json_text(dirty_json)
    assert result == "{\"key\": \"value\"}"