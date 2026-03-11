"""Shared HTTP/retry helpers with typed errors."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Type

import requests


class HttpClientError(RuntimeError):
    """Base error for shared HTTP client helpers."""


class HttpRequestError(HttpClientError):
    """Network-level request failure."""


class HttpResponseError(HttpClientError):
    """Non-success HTTP response."""


class HttpDecodeError(HttpClientError):
    """Response body could not be decoded."""


_DEFAULT_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _backoff_sleep(
    attempt: int,
    *,
    base: float,
    max_sleep: float,
    jitter_ratio: float,
) -> None:
    """Exponential backoff with bounded jitter."""
    delay = min(max_sleep, base * (2 ** max(0, attempt - 1)))
    if jitter_ratio > 0:
        jitter = delay * jitter_ratio
        delay = max(0.0, delay + random.uniform(-jitter, jitter))
    time.sleep(delay)


def request_with_retry(
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    data: Any = None,
    json_payload: Any = None,
    timeout: float = 10.0,
    retries: int = 3,
    backoff_base: float = 0.5,
    backoff_max: float = 8.0,
    jitter_ratio: float = 0.2,
    retry_statuses: Optional[Iterable[int]] = None,
    on_retry: Optional[Callable[[int, int, Exception], None]] = None,
    session: Optional[requests.Session] = None,
) -> requests.Response:
    """Perform an HTTP request with retry/backoff and typed failures."""
    retry_codes = set(retry_statuses or _DEFAULT_RETRY_STATUSES)
    requester = session.request if session is not None else requests.request
    attempts = max(1, int(retries))

    for attempt in range(1, attempts + 1):
        try:
            resp = requester(
                method=method,
                url=url,
                params=params,
                headers=headers,
                data=data,
                json=json_payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            wrapped = HttpRequestError(f"HTTP request failed for {method} {url}: {exc}")
            if attempt >= attempts:
                raise wrapped from exc
            if on_retry:
                on_retry(attempt, attempts, wrapped)
            _backoff_sleep(
                attempt,
                base=backoff_base,
                max_sleep=backoff_max,
                jitter_ratio=jitter_ratio,
            )
            continue

        if resp.status_code in retry_codes and attempt < attempts:
            wrapped = HttpResponseError(
                f"HTTP {resp.status_code} for {method} {url} (retry {attempt}/{attempts})"
            )
            if on_retry:
                on_retry(attempt, attempts, wrapped)
            _backoff_sleep(
                attempt,
                base=backoff_base,
                max_sleep=backoff_max,
                jitter_ratio=jitter_ratio,
            )
            continue

        if resp.status_code >= 400:
            snippet = (resp.text or "")[:300]
            raise HttpResponseError(
                f"HTTP {resp.status_code} for {method} {url}: {snippet}"
            )

        return resp

    raise HttpClientError(f"HTTP retries exhausted for {method} {url}")


def get_json(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 10.0,
    retries: int = 3,
    backoff_base: float = 0.5,
    backoff_max: float = 8.0,
    jitter_ratio: float = 0.2,
    retry_statuses: Optional[Iterable[int]] = None,
    on_retry: Optional[Callable[[int, int, Exception], None]] = None,
    session: Optional[requests.Session] = None,
) -> Any:
    """GET JSON with retry/backoff and typed decode errors."""
    resp = request_with_retry(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        retries=retries,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
        jitter_ratio=jitter_ratio,
        retry_statuses=retry_statuses,
        on_retry=on_retry,
        session=session,
    )
    try:
        return resp.json()
    except ValueError as exc:
        raise HttpDecodeError(f"JSON decode failed for GET {url}") from exc


def get_text(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 10.0,
    retries: int = 3,
    backoff_base: float = 0.5,
    backoff_max: float = 8.0,
    jitter_ratio: float = 0.2,
    retry_statuses: Optional[Iterable[int]] = None,
    on_retry: Optional[Callable[[int, int, Exception], None]] = None,
    session: Optional[requests.Session] = None,
) -> str:
    """GET text with retry/backoff."""
    resp = request_with_retry(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        retries=retries,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
        jitter_ratio=jitter_ratio,
        retry_statuses=retry_statuses,
        on_retry=on_retry,
        session=session,
    )
    return resp.text


def retry_call(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    backoff_base: float = 0.8,
    backoff_max: float = 8.0,
    jitter_ratio: float = 0.15,
    retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable[[int, int, BaseException], None]] = None,
    **kwargs: Any,
) -> Any:
    """Retry any callable with jittered exponential backoff."""
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except retry_exceptions as exc:
            if attempt >= attempts:
                raise
            if on_retry:
                on_retry(attempt, attempts, exc)
            _backoff_sleep(
                attempt,
                base=backoff_base,
                max_sleep=backoff_max,
                jitter_ratio=jitter_ratio,
            )
