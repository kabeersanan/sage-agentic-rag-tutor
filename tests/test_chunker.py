from src.ingestion.chunker import count_tokens, extract_topic_header

def test_count_tokens():
    text = "This is a simple test string to count tokens."
    tokens = count_tokens(text)
    assert tokens > 0
    assert tokens < 20  # It should be a small number

def test_extract_topic_header():
    text = "Chapter 1: Chemical Reactions\nThis chapter explains how substances react."
    header = extract_topic_header(text)
    # It should grab the first short line as the topic
    assert header == "Chapter 1: Chemical Reactions"

def test_extract_topic_header_fallback():
    text = "This is a very long sentence that just keeps going and going and doesn't really have a clear header at the top so it should fall back."
    header = extract_topic_header(text)
    assert header == "General Section"