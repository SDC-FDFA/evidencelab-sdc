"""OpenAI-compatible embedding client (for Infomaniak / OpenAI endpoints)."""

import logging
import os
from typing import Generator, Iterable, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient:
    """Client for OpenAI-compatible embedding endpoints (Infomaniak, OpenAI, etc.)."""

    def __init__(self, base_url: str, model_name: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("INFOMANIAK_API_KEY")
        self.request_timeout = int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "120"))

    def embed(
        self, documents: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        docs = list(documents) if not isinstance(documents, list) else documents
        if not docs:
            return

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Default batching
        bs = batch_size if batch_size and batch_size > 0 else 16
        for i in range(0, len(docs), bs):
            batch = docs[i : i + bs]
            try:
                payload = {"model": self.model_name, "input": batch}
                response = requests.post(
                    url, json=payload, headers=headers, timeout=self.request_timeout
                )
                if response.status_code != 200:
                    raise requests.exceptions.HTTPError(
                        f"{response.status_code} Client Error: {response.reason} for url: {url} | "
                        f"Response: {response.text}",
                        response=response,
                    )
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                for emb in embeddings:
                    yield np.array(emb, dtype=np.float32)
            except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
                logger.error(
                    "OpenAI embedding failed for batch of size %s: %s",
                    len(batch),
                    exc,
                )
                raise