import html
import re

# Regex for leading chat preambles from instruction-following LLMs
PREAMBLE_PATTERN = re.compile(
    r"^(?:(?:Sure|Certainly|Of course)[!,.]?\s*(?:Here(?:\'s| is)|I(?:\'d| would) be happy to|below is|this is)?|"
    r"(?:Here(?:\'s| is)|Below is)\s+(?:a|an|the|my)?\s*(?:AI|sample|generated|draft|possible)?\s*(?:IMDb\s+)?review\b)"
    r"[^:\n]{0,150}:\s*",
    re.IGNORECASE,
)

# HTML tags regex (e.g. <br />, <br>, <p>, </p>, etc.)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Unicode replacements
UNICODE_QUOTES_DASHES = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201a": "'",  # single low-9 quotation mark
    "\u201b": "'",  # single high-reversed-9 quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u201e": '"',  # double low-9 quotation mark
    "\u201f": '"',  # double high-reversed-9 quotation mark
    "\u00ab": '"',  # left-pointing double angle quotation mark
    "\u00bb": '"',  # right-pointing double angle quotation mark
    "\u2014": " - ",  # em dash
    "\u2013": " - ",  # en dash
    "\u2015": " - ",  # horizontal bar
    "\u2212": "-",  # minus sign
    "\u00a0": " ",  # non-breaking space
    "\u2026": "...",  # ellipsis
    "`": "'",
    "´": "'",
}


def clean_text(text: str) -> str:
    """
    Standard text cleaning pipeline for review text:
    1. Unescapes HTML entities (&amp;, &quot;, &#39;, etc.)
    2. Strips HTML tags (<br />, <p>, etc.)
    3. Normalizes unicode quotes, apostrophes, dashes, and spaces
    4. Removes leading chat preambles (e.g. 'Sure! Here's my review of ...:')
    5. Collapses multiple whitespace characters to a single space and strips.
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. Unescape HTML entities
    cleaned = html.unescape(text)

    # 2. Strip HTML tags (replace with space to prevent concatenating adjacent words)
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)

    # 3. Normalize unicode quotes and dashes
    for char, replacement in UNICODE_QUOTES_DASHES.items():
        if char in cleaned:
            cleaned = cleaned.replace(char, replacement)

    # 4. Remove leading chat preambles
    cleaned = PREAMBLE_PATTERN.sub("", cleaned)

    # 5. Collapse repeated whitespace and strip
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def has_html_br(text: str) -> bool:
    """Detect if text contains HTML br tags (<br />, <br>, <br/>)."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r"<br\s*/?>", text, re.IGNORECASE))


def has_chat_preamble(text: str) -> bool:
    """Detect if text contains a leading LLM chat preamble."""
    if not isinstance(text, str):
        return False
    return bool(PREAMBLE_PATTERN.match(text))

