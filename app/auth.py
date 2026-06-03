import secrets
import time

from fastapi import Security, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Brute force tracking: {ip: [timestamp, ...]}
_failed_attempts: dict[str, list[float]] = {}
_MAX_ATTEMPTS = 10
_WINDOW_SECONDS = 60
_BLOCK_SECONDS = 300


def _get_client_ip(request: Request) -> str:
    # Honor X-Forwarded-For only when configured to trust a proxy
    if settings.trust_proxy:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    if request.client is None:
        return "unknown"
    return request.client.host


def _check_brute_force(ip: str) -> None:
    now = time.time()
    attempts = _failed_attempts.get(ip, [])
    # Drop attempts outside the rolling window
    attempts = [t for t in attempts if now - t < _WINDOW_SECONDS]
    if attempts:
        _failed_attempts[ip] = attempts
    elif ip in _failed_attempts:
        del _failed_attempts[ip]

    if len(attempts) >= _MAX_ATTEMPTS:
        retry_after = int(_BLOCK_SECONDS - (now - attempts[0]))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed auth attempts. Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


def _record_failure(ip: str) -> None:
    attempts = _failed_attempts.get(ip, [])
    attempts.append(time.time())
    _failed_attempts[ip] = attempts


def verify_api_key(request: Request, key: str = Security(_api_key_header)) -> None:
    if not settings.api_key:
        if not settings.allow_no_auth:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server misconfigured: API_KEY is not set. Set ALLOW_NO_AUTH=true to disable auth explicitly.",
            )
        return

    ip = _get_client_ip(request)
    _check_brute_force(ip)

    # Constant-time comparison prevents timing attacks
    if not secrets.compare_digest(key or "", settings.api_key):
        _record_failure(ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    # Clear failure history on successful auth
    _failed_attempts.pop(ip, None)
