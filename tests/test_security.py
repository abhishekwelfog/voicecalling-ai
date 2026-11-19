"""Tests for phone normalization, order ownership verification, PBKDF2 hashing, and DDoS defense."""

import pytest
import hashlib
from app.utils.security import (
    normalize_indian_phone_number,
    is_valid_phone_number,
    verify_order_ownership,
    sanitize_csv_cell,
    clean_speech_transcript,
    strip_recording_announcement,
    is_voicemail_or_machine
)
from app.utils.pii_masking import mask_sensitive_pii
from app.services.security_service import (
    hash_password,
    verify_password,
    AntiBruteForceManager,
    LEGACY_PASSWORD_SALT
)
from app.middleware.security import SlidingWindowRateLimiter
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.auth_service import SESSION_COOKIE_NAME, create_session_token


def test_normalize_indian_phone():
    assert normalize_indian_phone_number("9876543210") == "919876543210"
    assert normalize_indian_phone_number("09876543210") == "919876543210"
    assert normalize_indian_phone_number("+91 9876543210") == "919876543210"
    assert normalize_indian_phone_number("919876543210") == "919876543210"


def test_is_valid_phone_number():
    assert is_valid_phone_number("9876543210") is True
    assert is_valid_phone_number("8123456789") is True
    assert is_valid_phone_number("12345") is False
    assert is_valid_phone_number("abcdef") is False


def test_verify_order_ownership():
    # Direct match on user_id in order details
    assert verify_order_ownership(
        requested_order_id="101",
        user_id=1,
        order_data={"id": "101", "user_id": 1}
    ) is True

    # User mismatch
    assert verify_order_ownership(
        requested_order_id="101",
        user_id=2,
        order_data={"id": "101", "user_id": 1}
    ) is False

    # Match in user's order history
    history = [
        {"id": "50", "grand_total": 500},
        {"id": "51", "grand_total": 800}
    ]
    assert verify_order_ownership("51", user_id=1, order_history=history) is True
    assert verify_order_ownership("99", user_id=1, order_history=history) is False


def test_pbkdf2_password_hashing_and_verification():
    raw_pwd = "SecretAdminPassword#2026!"
    hashed = hash_password(raw_pwd)

    # 1. Format verification
    assert hashed.startswith("pbkdf2_sha256$600000$")
    parts = hashed.split("$")
    assert len(parts) == 4
    assert len(parts[2]) == 32  # 16-byte salt in hex = 32 chars
    assert len(parts[3]) == 64  # SHA256 output = 64 hex chars

    # 2. Correct password verification
    is_valid, needs_rehash = verify_password(raw_pwd, hashed)
    assert is_valid is True
    assert needs_rehash is False

    # 3. Wrong password rejection
    is_wrong, _ = verify_password("WrongPassword123", hashed)
    assert is_wrong is False

    # 4. Salt randomness: two hashes of identical password must never match
    hashed_second = hash_password(raw_pwd)
    assert hashed != hashed_second


def test_legacy_sha256_hash_backward_compatibility():
    raw_pwd = "admin@123"
    legacy_hash = hashlib.sha256((LEGACY_PASSWORD_SALT + raw_pwd).encode("utf-8")).hexdigest()

    # Must verify legacy hash and flag needs_rehash=True for auto-upgrade
    is_valid, needs_rehash = verify_password(raw_pwd, legacy_hash)
    assert is_valid is True
    assert needs_rehash is True

    # Bad password with legacy hash
    is_wrong, _ = verify_password("incorrect_pwd", legacy_hash)
    assert is_wrong is False


def test_anti_brute_force_lockout():
    bf = AntiBruteForceManager(max_attempts=5, lockout_seconds=900)
    ip = "192.168.1.50"
    user = "attacker_target"

    # First 4 attempts fail -> not yet locked
    for i in range(4):
        remaining, is_locked = bf.record_failure(ip, user)
        assert is_locked is False
        assert remaining == 4 - i
        assert bf.check_lockout(ip, user) is None

    # 5th attempt fails -> triggers 15-min lockout
    remaining, is_locked = bf.record_failure(ip, user)
    assert is_locked is True
    assert remaining == 0
    lockout_sec = bf.check_lockout(ip, user)
    assert lockout_sec is not None
    assert lockout_sec > 800

    # Success clears lockout
    bf.record_success(ip, user)
    assert bf.check_lockout(ip, user) is None


def test_sliding_window_rate_limiter_ddos_protection():
    limiter = SlidingWindowRateLimiter()
    ip = "203.0.113.195"
    bucket = "test_bucket"

    # Allow up to 3 requests per 10 seconds
    for _ in range(3):
        allowed, remaining, retry_after = limiter.is_allowed(ip, bucket, max_requests=3, window_seconds=10)
        assert allowed is True

    # 4th request must be throttled with HTTP 429 semantics
    allowed, remaining, retry_after = limiter.is_allowed(ip, bucket, max_requests=3, window_seconds=10)
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0


def test_csv_formula_injection_prevention():
    """Verify that dangerous CSV formula characters (=, +, -, @, \t, \r) are safely prefixed with single quote."""
    assert sanitize_csv_cell("=cmd|' /C calc'!A0") == "'=cmd|' /C calc'!A0"
    assert sanitize_csv_cell("+12345") == "'+12345"
    assert sanitize_csv_cell("-2+3") == "'-2+3"
    assert sanitize_csv_cell("@SUM(1,2)") == "'@SUM(1,2)"
    assert sanitize_csv_cell("\tmalicious_tab") == "'\tmalicious_tab"
    assert sanitize_csv_cell("Normal Customer Name") == "Normal Customer Name"
    assert sanitize_csv_cell(None) == ""


def test_pii_masking_in_transcripts():
    """Verify sensitive customer PII (credit cards, CVV, OTPs, passwords) are automatically redacted."""
    raw_transcript = (
        "Customer: My card is 4111 2222 3333 4444 and cvv is 987, otp is 123456. "
        "Also my password is my_secret_pass!"
    )
    masked = mask_sensitive_pii(raw_transcript)
    assert "4111 2222 3333 4444" not in masked
    assert "987" not in masked
    assert "123456" not in masked


@pytest.mark.asyncio
async def test_unauthenticated_api_rejection():
    """Verify that protected API endpoints strictly reject unauthenticated requests with HTTP 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Protected endpoints without session token
        res_cust = await client.get("/api/customers")
        assert res_cust.status_code == 401

        res_stats = await client.get("/api/stats")
        assert res_stats.status_code == 401

        res_calls = await client.get("/api/call-logs")
        assert res_calls.status_code == 401

        res_webrtc = await client.get("/api/telecmi/webrtc-config")
        assert res_webrtc.status_code == 401

        res_chpass = await client.post("/api/auth/change-password", json={"current_password": "a", "new_password": "b"})
        assert res_chpass.status_code == 401


@pytest.mark.asyncio
async def test_security_headers_middleware():
    """Verify OWASP security headers are present in HTTP responses."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        headers = res.headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "SAMEORIGIN"
        assert headers.get("X-XSS-Protection") == "1; mode=block"
        assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert headers.get("Permissions-Policy") == "geolocation=(), camera=(), microphone=(self)"
        assert headers.get("Server") == "Welfog-Security-Shield"


@pytest.mark.asyncio
async def test_sql_injection_defense_on_search():
    """Verify SQL injection payloads in search query do not break database or leak unintended records."""
    token = create_session_token("admin")
    cookies = {SESSION_COOKIE_NAME: token}
    sqli_payloads = [
        "' OR '1'='1",
        "'; DROP TABLE customers; --",
        "1' UNION SELECT username, password_hash, NULL, NULL FROM admins --",
        "' OR 1=1 --",
    ]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", cookies=cookies) as client:
        for payload in sqli_payloads:
            res = await client.get(f"/api/customers?search={payload}")
            # Must return 200 with normal empty or filtered list, never 500 error or SQL crash
            assert res.status_code == 200
            data = res.json()
            assert "customers" in data
            assert isinstance(data["customers"], list)


def test_foreign_asr_hallucination_cleansing():
    """Verify that Japanese, Portuguese, and foreign script ASR hallucinations are cleansed."""
    # 1. Japanese thank you hallucination
    assert "どうも" not in clean_speech_transcript("どうもありがとう ございます。")
    assert clean_speech_transcript("どうもありがとう ございます。") in {"Haanji, thank you.", "Thank you."}

    # 2. Portuguese and Spanish noise hallucinations
    assert clean_speech_transcript("Sou eu, sim.") == "Haan, main hi hoon."
    assert clean_speech_transcript("Ah, sim.") == "Haanji."
    assert clean_speech_transcript("Sí.") == "Haanji."
    assert clean_speech_transcript("Tá.") == "Theek hai."

    # 3. Legitimate Indian customer speech (Hindi, Hinglish, English) remains 100% intact
    hindi_speech = "अच्छा तो अभी किया था मैंने वेलफोक्स वो वाला।"
    assert clean_speech_transcript(hindi_speech) == hindi_speech

    hinglish_speech = "Haan, kar do kar do."
    assert clean_speech_transcript(hinglish_speech) == hinglish_speech


def test_strip_recording_announcement():
    """Verify carrier/dialer automated call recording announcements are stripped."""
    assert strip_recording_announcement("This call is now being recorded. Hello, haanji boliye.") == "Hello, haanji boliye"
    assert strip_recording_announcement("This call is now being recorded.") == ""
    assert strip_recording_announcement("यह कॉल अब रिकॉर्ड की जा रही है। हां जी कौन बोल रहा है?") == "हां जी कौन बोल रहा है"
    assert strip_recording_announcement("Call recording started. Yes, Abhishek speaking.") == "Yes, Abhishek speaking"
    
    # Normal customer speech without recording announcement is 100% preserved
    speech = "Hello, main kal baat karunga."
    assert strip_recording_announcement(speech) == speech


def test_voicemail_detection():
    """Verify automated voicemail / answering machine detection (AMD)."""
    assert is_voicemail_or_machine("Please leave a message after the tone.") is True
    assert is_voicemail_or_machine("The person you are calling is not available to take your call, please leave your message after the beep.") is True
    assert is_voicemail_or_machine("Your call has been forwarded to voicemail.") is True
    assert is_voicemail_or_machine("The mailbox is full and cannot accept messages.") is True
    assert is_voicemail_or_machine("कृपया टोन के बाद अपना संदेश रिकॉर्ड करें।") is True
    assert is_voicemail_or_machine("The number you have dialed is currently switched off.") is True

    # Critical: Legitimate customer speech must NEVER trigger false positive AMD
    assert is_voicemail_or_machine("Haan, dispatch kar dijiye.") is False
    assert is_voicemail_or_machine("Hello, main Abhishek baat kar raha hoon.") is False
    assert is_voicemail_or_machine("Order ka kitna amount hai?") is False

