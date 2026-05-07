import time
from collections import defaultdict

from fastapi import Security, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Brute force tracking: {ip: [(timestamp, ...), ...]}
_failed_attempts: dict[str, list[float]] = defaultdict(list)
_MAX_ATTEMPTS = 10   # max failed attempts
_WINDOW_SECONDS = 60  # within this rolling window
_BLOCK_SECONDS = 300  # block duration after limit hit


def _check_brute_force(ip: str) -> None:
    now = time.time()
    attempts = _failed_attempts[ip]
    # Drop attempts outside the window
    _failed_attempts[ip] = [t for t in attempts if now - t < _WINDOW_SECONDS]
    if len(_failed_attempts[ip]) >= _MAX_ATTEMPTS:
        oldest = _failed_attempts[ip][0]
        retry_after = int(_BLOCK_SECONDS - (now - oldest))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed auth attempts. Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


def verify_api_key(request: Request, key: str = Security(_api_key_header)) -> None:
    if not settings.api_key:
        return  # auth disabled when API_KEY is not set (local dev)

    ip = request.client.host
    _check_brute_force(ip)

    if key != settings.api_key:
        _failed_attempts[ip].append(time.time())
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    # Successful auth clears the failed attempts for this IP
    _failed_attempts.pop(ip, None)
