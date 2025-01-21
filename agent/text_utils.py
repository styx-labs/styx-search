import re

def clean_text(text: str) -> str:
    """Clean text by removing special characters and converting to lowercase."""
    return re.sub(r"[^\w\s]", " ", text.lower())
