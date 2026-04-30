from app.preprocessing import preprocess_message

# Mojibake characters built from Unicode escapes to avoid file encoding issues
# U+2019 RIGHT SINGLE QUOTATION MARK -> UTF-8 \xe2\x80\x99 -> decoded as cp1252 -> a-circ + euro + trade
APOS   = "â€™"   # corrupted apostrophe / right-single-quote
EMDASH = "â€”"   # corrupted em-dash

# The fixed versions (what preprocessing should produce)
FIXED_APOS   = "’"   # U+2019 RIGHT SINGLE QUOTATION MARK
FIXED_EMDASH = "—"   # U+2014 EM DASH

SAMPLE = (
    "Hi Steve,<br><br>"
    "Thank you for getting in touch with the NOBL Customer Success Team. "
    f"My name is Ashley, your AI Travel Concierge. We{APOS}re here to make every step "
    f"simple, seamless, and hassle-free {EMDASH} and I{APOS}ll be happy to assist you today, "
    "regarding your order not arriving.<br><br>"
    f"I{APOS}m sorry you haven{APOS}t received your order yet. "
    "I'm connecting you with our specialist team right now so they can locate your order "
    "and provide you with an update.<br><br>"
    f"I{APOS}ve transferred your case to our Customer Success team. "
    f"They will follow up with you within 24{EMDASH}48 hours to ensure this is fully resolved."
)


def test_apostrophe_mojibake_fixed():
    result = preprocess_message(f"We{APOS}re here")
    assert APOS not in result
    assert f"We{FIXED_APOS}re here" in result


def test_em_dash_mojibake_fixed():
    result = preprocess_message(f"24{EMDASH}48 hours")
    assert EMDASH not in result
    assert f"24{FIXED_EMDASH}48 hours" in result


def test_br_replaced_with_newline():
    result = preprocess_message("line one<br>line two<br/>line three<BR>line four")
    assert "<br" not in result.lower()
    assert "line one\nline two\nline three\nline four" in result


def test_double_br_becomes_double_newline():
    result = preprocess_message("para one<br><br>para two")
    assert "para one\n\npara two" in result


def test_html_tags_stripped():
    result = preprocess_message("<p>Hello <b>world</b></p>")
    assert result == "Hello world"


def test_html_entities_amp_decoded():
    result = preprocess_message("Tom &amp; Jerry")
    assert result == "Tom & Jerry"


def test_html_entities_numeric_decoded():
    result = preprocess_message("say &#39;hello&#39;")
    assert result == "say 'hello'"


def test_escaped_angle_brackets_stripped():
    # &lt;x&gt; unescapes to <x> which is then stripped as an HTML tag
    result = preprocess_message("before &lt;tag&gt; after")
    assert "&lt;" not in result
    assert "<tag>" not in result
    assert "before" in result
    assert "after" in result


def test_excessive_blank_lines_collapsed():
    result = preprocess_message("a\n\n\n\n\nb")
    assert "\n\n\n" not in result
    assert "a\n\nb" in result


def test_leading_trailing_whitespace_stripped():
    result = preprocess_message("  hello  ")
    assert result == "hello"


def test_plain_text_unchanged():
    result = preprocess_message("Hello, this is a normal message.")
    assert result == "Hello, this is a normal message."


def test_full_sample_no_mojibake():
    result = preprocess_message(SAMPLE)
    assert APOS not in result
    assert EMDASH not in result


def test_full_sample_no_html():
    result = preprocess_message(SAMPLE)
    assert "<br>" not in result
    assert "<br/>" not in result


def test_full_sample_content_preserved():
    result = preprocess_message(SAMPLE)
    assert "Hi Steve," in result
    assert "Ashley" in result
    assert f"We{FIXED_APOS}re" in result
    assert f"24{FIXED_EMDASH}48 hours" in result
