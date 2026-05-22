# tests/test_data.py
import pytest
from madewithml.data import clean_text
from madewithml.config import STOPWORDS

def test_clean_text():
    """Ensure the text cleaning function handles casing, symbols, and links."""
    raw_text = "Check out this AWESOME project! It uses React. https://example.com"
    cleaned = clean_text(raw_text, stopwords=[])
    
    assert "https" not in cleaned
    assert "awesome" in cleaned
    assert "!" not in cleaned

def test_stopwords_removal():
    """Ensure defined stopwords are completely removed from the text."""
    raw_text = "I am building a model for our team"
    # Assuming "i", "am", "a", "for", "our" are in the STOPWORDS list
    cleaned = clean_text(raw_text, stopwords=STOPWORDS)
    
    assert " i " not in f" {cleaned} "
    assert " our " not in f" {cleaned} "
    assert "model" in cleaned
    assert "team" in cleaned