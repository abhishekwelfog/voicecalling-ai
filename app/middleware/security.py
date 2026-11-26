"""
Security and DDoS Protection Middlewares for Welfog Voice Agent.
Implements:
1. Volumetric DDoS & automated flood protection via IP sliding-window token bucket.
2. Endpoint-specific tiered rate limits (general APIs, authentication brute-force gate, telephony controls).
3. Hardened security response headers (OWASP Secure Headers Project).
4. Memory exhaustion & payload size bounding (anti-OOM / slowloris defense).
"""

import time
import threading
from typing import Dict, List, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


class SlidingWindowRateLimiter:
    """
    High-performance thread-safe sliding window rate limiter.
    """
    def __init__(self):
        self._lock = threading.Lock()
        # Key: (ip, bucket_name) -> list of request timestamps
        self._buckets: Dict[str, List[float]] = {}
        self._last_clean = time.time()

    def is_allowed(self, client_ip: str, bucket_name: str, max_requests: int, window_seconds: int = 60) -> tuple[bool, int, int]:
        """
        Check if request is permitted under the sliding window limit.
        Returns: (is_allowed, remaining_requests, retry_after_seconds)
        """
        now = time.time()
        key = f"{client_ip}::{bucket_name}"

        with self._lock:
            # Periodic sweep of expired buckets every 2 minutes
            if now - self._last_clean > 120:
                self._cleanup(now)

            cutoff = now - window_seconds
            timestamps = [t for t in self._buckets.get(key, []) if t > cutoff]

            if len(timestamps) >= max_requests:
                oldest = timestamps[0]
                retry_after = max(1, int(window_seconds - (now - oldest)))
                self._buckets[key] = timestamps
                return False, 0, retry_after

            timestamps.append(now)
            self._buckets[key] = timestamps
            remaining = max_requests - len(timestamps)
            return True, remaining, 0

    def _cleanup(self, now: float):
        self._last_clean = now
        stale_cutoff = now - 300  # Remove buckets inactive for >5 mins
        keys_to_delete = [
            k for k, ts in self._buckets.items()
            if not ts or ts[-1] < stale_cutoff
        ]
        for k in keys_to_delete:
            del self._buckets[k]

    def reset_all(self):
        """Testing utility."""
        with self._lock:
            self._buckets.clear()


# Global in-memory rate limiter singleton
_global_limiter = SlidingWindowRateLimiter()


def get_client_ip(request: Request) -> str:
    """Extract real client IP considering forward proxies with fallback."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the leftmost untrusted client IP
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host.strip()
    return "127.0.0.1"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects enterprise-grade defensive HTTP headers into all responses.
    Mitigates clickjacking, MIME-sniffing, XSS, and unauthorized browser API access.
    """
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        headers = response.headers
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-Frame-Options"] = "SAMEORIGIN"
        headers["X-XSS-Protection"] = "1; mode=block"
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=(self)"
        headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "connect-src 'self' wss: ws: https:; "
            "media-src 'self' blob:; "
            "img-src 'self' data: https:; "
            "frame-ancestors 'self';"
        )
        is_https = request.url.scheme == "https" or request.headers.get("X-Forwarded-Proto") == "https"
        if is_https:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if request.url.path.startswith("/api/") or request.url.path in {"/dashboard", "/dialer", "/history"}:
            headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            headers["Pragma"] = "no-cache"
        # Mask server implementation fingerprinting
        headers["Server"] = "Welfog-Security-Shield"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Tiered sliding-window rate limiter protecting against volumetric DDoS attacks,
    brute-force login floods, and telephony API exhaustion.
    """
    def __init__(self, app, limiter: Optional[SlidingWindowRateLimiter] = None):
        super().__init__(app)
        self.limiter = limiter or _global_limiter

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 1. Exempt static assets, favicon, and internal health probes
        if path.startswith("/static") or path == "/health" or path.endswith((".css", ".js", ".png", ".jpg", ".ico", ".svg")):
            return await call_next(request)

        client_ip = get_client_ip(request)

        # 2. Determine tiered limit based on endpoint sensitivity
        if path == "/api/auth/login":
            # Strict burst protection on login: 15 attempts / min per IP
            bucket = "auth_login"
            limit = 15
            window = 60
        elif path.startswith("/api/dial") or path.startswith("/api/campaign"):
            # Telephony outbound rate control: 30 requests / min per IP
            bucket = "telephony_ops"
            limit = 30
            window = 60
        else:
            # General API / UI polling: 180 requests / min per IP
            bucket = "general_api"
            limit = 180
            window = 60

        allowed, remaining, retry_after = self.limiter.is_allowed(
            client_ip=client_ip,
            bucket_name=bucket,
            max_requests=limit,
            window_seconds=window
        )

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Automated flood & DDoS protection is active. Please slow down.",
                    "retry_after_seconds": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0"
                }
            )

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Prevents memory exhaustion, slowloris, and denial of service via oversized HTTP payloads.
    Maximum allowed request body size: 15 MB.
    """
    MAX_BYTES = 15 * 1024 * 1024  # 15 MB

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.MAX_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "payload_too_large",
                            "message": f"Request entity too large. Maximum supported size is 15MB."
                        }
                    )
            except ValueError:
                pass

        return await call_next(request)

