"""Shared HTTP/retry helpers with typed errors."""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Type

import requests

logger = logging.getLogger(__name__)

_TRUTHY_ENV = {"1", "true", "yes", "on"}
_CA_BUNDLE_ENV = "NBA_HTTP_CA_BUNDLE"
_ALLOW_INSECURE_SSL_ENV = "NBA_HTTP_ALLOW_INSECURE_SSL"

try:
    import certifi

    _CERTIFI_CA_BUNDLE = certifi.where()
except Exception:
    _CERTIFI_CA_BUNDLE = None


class HttpClientError(RuntimeError):
    """Base error for shared HTTP client helpers."""


class HttpRequestError(HttpClientError):
    """Network-level request failure."""


class HttpResponseError(HttpClientError):
    """Non-success HTTP response."""


class HttpDecodeError(HttpClientError):
    """Response body could not be decoded."""


_DEFAULT_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV


def _resolve_verify(verify: Optional[Any]) -> Any:
    """Resolve TLS verify setting with secure defaults."""
    if verify is not None:
        return verify

    ca_bundle = os.environ.get(_CA_BUNDLE_ENV, "").strip()
    if ca_bundle:
        return ca_bundle

    if _CERTIFI_CA_BUNDLE:
        return _CERTIFI_CA_BUNDLE

    return True


def _is_certificate_verify_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "certificate verify failed",
            "self-signed certificate",
            "self signed certificate",
            "unable to get local issuer certificate",
            "hostname mismatch",
        )
    )


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
    verify: Optional[Any] = None,
) -> requests.Response:
    """Perform an HTTP request with retry/backoff and typed failures."""
    retry_codes = set(retry_statuses or _DEFAULT_RETRY_STATUSES)
    requester = session.request if session is not None else requests.request
    attempts = max(1, int(retries))
    resolved_verify = _resolve_verify(verify)
    allow_insecure_ssl = _is_truthy_env(_ALLOW_INSECURE_SSL_ENV)
    used_insecure_fallback = False

    attempt = 1
    while attempt <= attempts:
        try:
            resp = requester(
                method=method,
                url=url,
                params=params,
                headers=headers,
                data=data,
                json=json_payload,
                timeout=timeout,
                verify=resolved_verify,
            )
        except requests.RequestException as exc:
            if (
                allow_insecure_ssl
                and not used_insecure_fallback
                and isinstance(exc, requests.exceptions.SSLError)
                and _is_certificate_verify_error(exc)
                and resolved_verify is not False
            ):
                used_insecure_fallback = True
                resolved_verify = False
                logger.warning(
                    "TLS verify failed for %s %s; retrying once with verify=False "
                    "because %s is enabled.",
                    method,
                    url,
                    _ALLOW_INSECURE_SSL_ENV,
                )
                if attempt >= attempts:
                    attempts += 1
                attempt += 1
                continue

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
            attempt += 1
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
            attempt += 1
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
    verify: Optional[Any] = None,
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
        verify=verify,
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
    verify: Optional[Any] = None,
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
        verify=verify,
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
