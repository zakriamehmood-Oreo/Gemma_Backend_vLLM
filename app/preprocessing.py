import html
import re
import unicodedata


def preprocess_message(text: str) -> str:
    # Fix mojibake — text that was UTF-8 but got decoded as latin-1
    # e.g. â€™ -> ' and â€" -> —
    try:
        text = text.encode("cp1252").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass  # already valid encoding, leave as-is

    # Decode HTML entities: &amp; -> &, &lt; -> <, &#39; -> ', etc.
    text = html.unescape(text)

    # Replace <br> and <br/> with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize unicode: NFKC converts lookalike characters to canonical form
    # e.g. ﬁ → fi, ² → 2, ＡＢＣ → ABC, …→ ...
    text = unicodedata.normalize("NFKC", text)

    # Normalize whitespace: tabs and non-breaking spaces to regular spaces
    text = re.sub(r"[\t\xa0 ​‌‍﻿]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


_REPLY_THREAD_RE = re.compile(
    r"\nOn\s+[\s\S]{0,300}?wrote:\s*\n[\s\S]*",
    re.IGNORECASE,
)

_SIG_LINE_PATTERNS = [
    re.compile(r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$"),
    re.compile(r"^(?:work|cell|tel|phone|fax|mobile)[\s:]+[\d\s\-()+\.]+$", re.IGNORECASE),
    re.compile(r"^\+?[\d\s\-()+\.]{7,20}$"),
    re.compile(r"^sent from (?:my )?\w[\w\s]*$", re.IGNORECASE),
    re.compile(
        r"^[\w\s]+"
        r"(?:[,\s]+(?:M\.?D\.?|Ph\.?D\.?|D\.?O\.?|PT|DPT|RN|NP|PA|MBA|Esq\.?))+"
        r"[\s,\.]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[\w\s\-&,\.]+(?:hospital|clinic|therapy|therapies|medical|health|"
        r"center|centre|institute|university|college|group|associates|"
        r"inc\.?|llc\.?|ltd\.?|corp\.?)[\s,]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:physical|occupational|speech|respiratory|cardiac|orthopedic|"
        r"pediatric|geriatric|sports|aquatic)?\s*"
        r"(?:therapist|physician|surgeon|nurse|doctor|specialist|consultant|"
        r"coordinator|manager|director|administrator|assistant|technician|"
        r"analyst|engineer|developer|designer|representative|agent)s?[\s,]*$",
        re.IGNORECASE,
    ),
]

_NOISE_LINE_PATTERNS = [
    re.compile(r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$"),
    re.compile(r"^\+?[\d\s\-()+\.]{7,20}$"),
    re.compile(
        r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+\d{1,2}\s*(?:st|nd|rd|th)?\s*(?:\d{4})?$",
        re.IGNORECASE,
    ),
    re.compile(r"^https?://\S+$"),
]


def _strip_reply_thread(text: str) -> str:
    text = _REPLY_THREAD_RE.sub("", text)
    lines = [l for l in text.split("\n") if not l.strip().startswith(">")]
    return "\n".join(lines)


def _strip_signature(text: str) -> str:
    lines = text.split("\n")
    cut = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            cut = i
            continue
        if any(p.match(stripped) for p in _SIG_LINE_PATTERNS):
            cut = i
        else:
            break
    return "\n".join(lines[:cut]).strip()


def clean_ticket_message(text: str) -> str:
    """Full pipeline for ticket messages: mojibake → HTML → reply thread → signature."""
    text = preprocess_message(text)
    text = _strip_reply_thread(text)
    text = _strip_signature(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_incomplete_input(text: str) -> bool:
    """True if the text has no meaningful customer message (only metadata)."""
    stripped = text.strip()
    if len(stripped) < 15:
        return True
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if not lines:
        return True
    meaningful = [l for l in lines if not any(p.match(l) for p in _NOISE_LINE_PATTERNS)]
    return len(meaningful) == 0


def truncate_for_model(text: str, tokenizer, max_tokens: int = 512) -> str:
    """Truncate to token limit, preserving whole words."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
