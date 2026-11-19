"""
Enterprise Security Service for Welfog Voice Calling Agent.
Implements:
1. NIST & OWASP SP 800-63B compliant PBKDF2-HMAC-SHA256 password hashing (600,000 iterations).
2. Per-user dynamic cryptographic salt (16 bytes) against rainbow tables and GPU cracking.
3. Constant-time verification (hmac.compare_digest) against timing attacks.
4. Transparent legacy hash migration for seamless zero-downtime upgrades.
5. In-memory sliding-window brute-force and credential stuffing prevention with IP/username lockout.
"""

import hmac
import hashlib
import secrets
import time
import threading
from typing import Optional, Tuple, Dict


# Legacy salt retained strictly for transparent one-way migration of existing hashes
LEGACY_PASSWORD_SALT = "welfog_salt_2026"
PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ROUNDS = 600000  # OWASP 2023/2024 recommended minimum for PBKDF2-SHA256


def hash_password(password: str) -> str:
    """
    Hash password using PBKDF2-HMAC-SHA256 with a unique 16-byte cryptographic salt and 600,000 rounds.
    Output format: pbkdf2_sha256$<rounds>$<salt_hex>$<hash_hex>
    """
    salt_bytes = secrets.token_bytes(16)
    salt_hex = salt_bytes.hex()
    key_bytes = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=password.encode("utf-8"),
        salt=salt_bytes,
        iterations=PBKDF2_ROUNDS
    )
    return f"{PBKDF2_ALGORITHM}${PBKDF2_ROUNDS}${salt_hex}${key_bytes.hex()}"


def verify_password(password: str, stored_hash: str) -> Tuple[bool, bool]:
    """
    Verify password against stored hash using constant-time comparison.
    Returns: (is_valid, needs_rehash)
    - is_valid: True if password matches stored hash.
    - needs_rehash: True if hash was verified using legacy single-round SHA-256 and should be upgraded.
    """
    if not password or not stored_hash:
        return False, False

    stored_hash = stored_hash.strip()

    # Modern PBKDF2-HMAC-SHA256 format
    if stored_hash.startswith(f"{PBKDF2_ALGORITHM}$"):
        try:
            parts = stored_hash.split("$")
            if len(parts) != 4:
                return False, False
            _, rounds_str, salt_hex, expected_hex = parts
            rounds = int(rounds_str)
            salt_bytes = bytes.fromhex(salt_hex)

            candidate_bytes = hashlib.pbkdf2_hmac(
                hash_name="sha256",
                password=password.encode("utf-8"),
                salt=salt_bytes,
                iterations=rounds
            )
            is_valid = hmac.compare_digest(candidate_bytes.hex(), expected_hex)
            # Rehash needed if rounds were increased in settings
            needs_rehash = is_valid and (rounds < PBKDF2_ROUNDS)
            return is_valid, needs_rehash
        except Exception:
            return False, False

    # Legacy static-salt single-round SHA-256 backward compatibility
    try:
        legacy_candidate = hashlib.sha256((LEGACY_PASSWORD_SALT + password).encode("utf-8")).hexdigest()
        if hmac.compare_digest(legacy_candidate, stored_hash):
            return True, True  # Valid, must be rehashed to PBKDF2!
    except Exception:
        pass

    return False, False


class AntiBruteForceManager:
    """
    In-memory thread-safe rate-limiter and lockout tracker for authentication endpoints.
    Protects against automated credential stuffing, dictionary attacks, and password spraying.
    """
    def __init__(self, max_attempts: int = 5, lockout_seconds: int = 900):
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_seconds
        self._lock = threading.Lock()
        # Key: (ip, username) -> list of failure timestamps
        self._failures: Dict[str, list] = {}
        # Key: (ip, username) -> locked_until_timestamp
        self._locks: Dict[str, float] = {}

    def _key(self, ip: str, username: str) -> str:
        clean_ip = (ip or "unknown").strip().lower()
        clean_user = (username or "global").strip().lower()
        return f"{clean_ip}::{clean_user}"

    def check_lockout(self, ip: str, username: str) -> Optional[int]:
        """
        Check if the IP/username pair is currently locked out.
        Returns:
            remaining_seconds (int) if locked out, or None if permitted.
        """
        now = time.time()
        key = self._key(ip, username)
        with self._lock:
            # Check individual account lock
            locked_until = self._locks.get(key, 0)
            if locked_until > now:
                return int(locked_until - now)

            # Check IP-wide lock (prevents spraying across usernames)
            ip_key = f"{ip.strip().lower()}::*"
            ip_locked_until = self._locks.get(ip_key, 0)
            if ip_locked_until > now:
                return int(ip_locked_until - now)

            # Clean expired locks
            if key in self._locks:
                del self._locks[key]
            if ip_key in self._locks:
                del self._locks[ip_key]

            return None

    def record_failure(self, ip: str, username: str) -> Tuple[int, bool]:
        """
        Record a failed login attempt.
        Returns:
            (remaining_attempts, is_now_locked)
        """
        now = time.time()
        key = self._key(ip, username)
        ip_key = f"{ip.strip().lower()}::*"

        with self._lock:
            # Filter timestamps within sliding window
            cutoff = now - self.lockout_seconds
            user_history = [t for t in self._failures.get(key, []) if t > cutoff]
            user_history.append(now)
            self._failures[key] = user_history

            ip_history = [t for t in self._failures.get(ip_key, []) if t > cutoff]
            ip_history.append(now)
            self._failures[ip_key] = ip_history

            # Account or IP lockout trigger
            if len(user_history) >= self.max_attempts:
                self._locks[key] = now + self.lockout_seconds
                self._failures[key] = []
                return 0, True

            if len(ip_history) >= (self.max_attempts * 3):
                # IP-wide spray lockout
                self._locks[ip_key] = now + self.lockout_seconds
                self._failures[ip_key] = []
                return 0, True

            remaining = max(0, self.max_attempts - len(user_history))
            return remaining, False

    def record_success(self, ip: str, username: str):
        """Reset failed attempt counters upon verified successful login."""
        key = self._key(ip, username)
        with self._lock:
            self._failures.pop(key, None)
            self._locks.pop(key, None)

    def reset_all(self):
        """Testing utility to clear all lockouts and failure counters."""
        with self._lock:
            self._failures.clear()
            self._locks.clear()


# Global Singleton
_brute_force_manager = AntiBruteForceManager(max_attempts=5, lockout_seconds=900)


def get_brute_force_manager() -> AntiBruteForceManager:
    return _brute_force_manager

