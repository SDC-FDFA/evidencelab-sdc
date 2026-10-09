"""Infomaniak reranker (OpenAI/Cohere-style ``/rerank`` endpoint)."""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ui.backend.services.google_vertex_reranker import RerankerUnavailableError

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30


def _get_endpoint(config: Dict[str, Any]) -> str:
    """Resolve the rerank URL.

    Order: model config ``endpoint_url`` > INFOMANIAK_RERANK_URL > COHERE_BASE_URL
    (e.g. https://api.infomaniak.com/2/ai/<id>/cohere/v2). ``/rerank`` is appended
    if missing.
    """
    endpoint = (
        config.get("endpoint_url")
        or os.getenv("INFOMANIAK_RERANK_URL")
        or os.getenv("COHERE_BASE_URL")
    )
    if not endpoint:
        raise ValueError(
            "Infomaniak rerank endpoint not configured: set COHERE_BASE_URL in "
            ".env (or 'endpoint_url' in the rerank model config)."
        )
    if not endpoint.startswith("https://"):
        raise ValueError(f"Infomaniak endpoint must use HTTPS, got: {endpoint!r}")
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/rerank"):
        endpoint += "/rerank"
    return endpoint


def _get_api_key() -> str:
    key = (
        os.getenv("COHERE_API_KEY")
        or os.getenv("INFOMANIAK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not key:
        raise ValueError(
            "COHERE_API_KEY (or INFOMANIAK_API_KEY) is required for reranking."
        )
    return key


def parse_rerank_response(data: Any, doc_count: int) -> List[float]:
    """Return scores in the original document order."""
    results: Optional[Any] = None
    if isinstance(data, dict):
        results = data.get("results") or data.get("data") or data.get("scores")
    if not isinstance(results, list) or not results:
        raise ValueError("Infomaniak rerank response missing results.")
    if all(isinstance(r, (int, float)) for r in results):
        if len(results) != doc_count:
            raise ValueError("Infomaniak rerank returned wrong number of scores.")
        return [float(r) for r in results]

    scores: List[Optional[float]] = [None] * doc_count
    for pos, item in enumerate(results):
        idx = item.get("index", pos)
        score = item.get("relevance_score")
        if score is None:
            score = item.get("score")
        if not isinstance(idx, int) or not 0 <= idx < doc_count or score is None:
            raise ValueError("Infomaniak rerank response has invalid entries.")
        scores[idx] = float(score)
    # Documents the API chose not to return get the lowest score.
    return [s if s is not None else float("-inf") for s in scores]


def rerank_with_infomaniak(
    query: str,
    documents: List[str],
    model_id: str,
    config: Dict[str, Any],
) -> List[float]:
    """Rerank via Infomaniak. Raises RerankerUnavailableError on transient errors."""
    payload = {
        "model": model_id,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
    }
    request = Request(
        _get_endpoint(config),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_get_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        if exc.code == 429 or exc.code >= 500:
            raise RerankerUnavailableError(
                f"Infomaniak rerank unavailable (HTTP {exc.code}): {body}"
            ) from exc
        raise ValueError(f"Infomaniak rerank failed (HTTP {exc.code}): {body}") from exc
    except (URLError, TimeoutError) as exc:
        raise RerankerUnavailableError(f"Infomaniak rerank unreachable: {exc}") from exc
    return parse_rerank_response(data, len(documents))