"""Security, validation, and normalization helpers."""

import re
from typing import Any, Dict, List, Optional


def normalize_indian_phone_number(phone: str) -> str:
    """
    Normalizes an Indian phone number to 91XXXXXXXXXX format as required by TeleCMI.
    Examples:
        '9876543210' -> '919876543210'
        '+91 9876543210' -> '919876543210'
        '09876543210' -> '919876543210'
        '919876543210' -> '919876543210'
    """
    # Remove all non-digit characters
    cleaned = re.sub(r"\D", "", phone or "")

    if len(cleaned) == 10:
        return f"91{cleaned}"
    elif len(cleaned) == 11 and cleaned.startswith("0"):
        return f"91{cleaned[1:]}"
    elif len(cleaned) == 12 and cleaned.startswith("91"):
        return cleaned
    elif len(cleaned) > 10 and cleaned.endswith(cleaned[-10:]):
        # Extract last 10 digits if country code is included
        return f"91{cleaned[-10:]}"
    return cleaned


def is_valid_phone_number(phone: str) -> bool:
    """Validate if phone number is a valid 12-digit Indian number starting with 91."""
    normalized = normalize_indian_phone_number(phone)
    return bool(re.match(r"^91[6-9]\d{9}$", normalized))


def sanitize_csv_cell(val: Any) -> str:
    """
    Sanitize CSV field to neutralize CSV / Formula Injection (CWE-1236).
    If a string starts with '=', '+', '-', '@', '\t', or '\r', prefix with single quote `'`
    so Excel and Google Sheets treat it strictly as safe text instead of executing formulas.
    """
    if val is None:
        return ""
    s = str(val)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{s}"
    return s


def sanitize_customer_output(text: str) -> str:
    """Strip any internal technical leaks or unwanted formatting before speech."""
    if not text:
        return ""
    # Strip json or code blocks if Gemini accidentally outputs them
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")
    return text.strip()


_CJK_JAPANESE_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")
_CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04ff]")
_ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff]")

# Android / Samsung / Google Phone dialer automated call recording announcements
_RECORDING_ANNOUNCEMENT_PATTERN = re.compile(
    r"(?i)(?:^|(?<=\s)|(?<=[.,!?-]))(?:this\s+call\s+(?:is\s+now\s+being|is\s+being|is|has\s+been)\s+recorded|"
    r"call\s+recording\s+(?:has\s+started|started|active|has\s+stopped|stopped)|"
    r"यह\s+कॉल\s+(?:अब\s+)?रिकॉर्ड\s+(?:की\s+जा\s+रही\s+है|हो\s+रही\s+है)|"
    r"कॉल\s+रिकॉर्डिंग\s+(?:शुरू\s+हो\s+गई\s+है|सक्रिय\s+है))[।.\s,!?-]*"
)

# Automated Voicemail / Answering Machine / Carrier IVR prompts
_VOICEMAIL_PATTERNS = [
    re.compile(r"(?i)\b(?:leave\s+(?:a|your)?\s*message\s+after\s+the\s+(?:tone|beep))\b"),
    re.compile(r"(?i)\b(?:record\s+your\s+message\s+after\s+the\s+(?:tone|beep))\b"),
    re.compile(r"(?i)\b(?:forwarded\s+to\s+voicemail|mailbox\s+is\s+full|not\s+available\s+to\s+take\s+your\s+call)\b"),
    re.compile(r"(?i)\b(?:टोन\s+के\s+बाद\s+(?:अपना\s+)?संदेश|वॉइस\s*मेल|कॉल\s+वॉइसमेल\s+पर)\b"),
    re.compile(r"(?i)\b(?:at\s+the\s+tone\s+please\s+record|please\s+leave\s+a\s+message)\b"),
    re.compile(r"(?i)\b(?:number\s+you\s+have\s+dialed\s+is\s+currently\s+busy|switched\s+off|out\s+of\s+coverage)\b"),
]


def strip_recording_announcement(text: str) -> str:
    """
    Strips automated Android/Samsung/Google Phone dialer recording announcements.
    Example: 'This call is now being recorded. Hello, haanji boliye.' -> 'Hello, haanji boliye.'
    """
    if not text:
        return ""
    if not _RECORDING_ANNOUNCEMENT_PATTERN.search(text):
        return text
    stripped = _RECORDING_ANNOUNCEMENT_PATTERN.sub("", text).strip(" ।.!?,")
    return stripped


def is_voicemail_or_machine(text: str) -> bool:
    """
    Detects whether an incoming utterance matches automated voicemail or answering machine prompts.
    """
    if not text:
        return False
    return any(p.search(text) for p in _VOICEMAIL_PATTERNS)

_ASR_HALLUCINATIONS = {
    "どうもありがとう ございます": "Haanji, thank you.",
    "どうもありがとうございます": "Haanji, thank you.",
    "どうもありがとう": "Thank you.",
    "ありがとうございます": "Dhanyawaad, thank you.",
    "ありがとう": "Thank you.",
    "はい": "Haanji.",
    "そうです": "Sahi hai.",
    "sou eu, sim": "Haan, main hi hoon.",
    "sou eu": "Main hi hoon.",
    "muito obrigado": "Bahut bahut dhanyawaad.",
    "obrigado": "Dhanyawaad.",
    "ah, sim": "Haanji.",
    "sim": "Haanji.",
    "sí": "Haanji.",
    "tá": "Theek hai.",
}


def clean_speech_transcript(text: Optional[str]) -> str:
    """
    Sanitize real-time audio transcriptions from Gemini Live ASR.
    - Strips automated phone dialer call-recording announcements.
    - Neutralizes universal multilingual ASR hallucinations (e.g. Japanese Hiragana/Katakana,
      Portuguese, Spanish) caused by faint telephony noise or compression, normalizing
      them to natural Indian conversational speech equivalents.
    """
    if not text:
        return ""
    cleaned = text.strip()

    # Strip automated carrier / phone dialer recording announcement if present
    stripped_rec = strip_recording_announcement(cleaned)
    if not stripped_rec:
        return ""
    cleaned = stripped_rec

    # Normalize punctuation and whitespace for lookup
    normalized_lookup = cleaned.lower().replace("  ", " ").strip(" .!?,。\u3002\uff0c\uff01\uff1f")
    if normalized_lookup in _ASR_HALLUCINATIONS:
        return _ASR_HALLUCINATIONS[normalized_lookup]

    for phrase, replacement in _ASR_HALLUCINATIONS.items():
        if phrase in cleaned or phrase in normalized_lookup:
            cleaned = re.sub(re.escape(phrase), replacement, cleaned, flags=re.IGNORECASE)

    # Clean foreign Japanese/Chinese/Korean or Cyrillic characters if any remain
    if _CJK_JAPANESE_PATTERN.search(cleaned) or _CYRILLIC_PATTERN.search(cleaned) or _ARABIC_PATTERN.search(cleaned):
        stripped = _CJK_JAPANESE_PATTERN.sub("", cleaned)
        stripped = _CYRILLIC_PATTERN.sub("", stripped)
        stripped = _ARABIC_PATTERN.sub("", stripped).strip(" .!?,\u3002\uff0c\uff01\uff1f")
        if stripped:
            return stripped
        # Entire utterance was a foreign speech hallucination (e.g. Japanese thank you)
        return "Haanji, thank you."

    # Clean trailing Japanese full stop if present
    if cleaned.endswith("。") or cleaned.endswith("\u3002"):
        cleaned = cleaned[:-1] + "."

    return cleaned


def verify_order_ownership(
    requested_order_id: str,
    user_id: int,
    order_data: Optional[Dict[str, Any]] = None,
    order_history: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Validates that a given order belongs to the requesting user_id.
    Prevents BOLA/IDOR attacks where customer A requests invoice or refund for customer B.
    """
    requested_str = str(requested_order_id).strip()

    # Direct match from order details if user_id or customer_id is present
    if order_data:
        # Check standard fields: user_id, customer_id, user.id
        owner_id = order_data.get("user_id") or order_data.get("customer_id")
        if owner_id is not None and int(owner_id) == int(user_id):
            return True
        user_obj = order_data.get("user")
        if isinstance(user_obj, dict) and user_obj.get("id") is not None:
            if int(user_obj["id"]) == int(user_id):
                return True

    # Check within user's order history list
    if order_history:
        for order in order_history:
            oid = str(order.get("id") or order.get("oid") or order.get("order_id") or order.get("code") or "")
            if oid and oid == requested_str:
                return True

    return False
