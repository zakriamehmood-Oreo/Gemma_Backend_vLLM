import html
import re


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

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
