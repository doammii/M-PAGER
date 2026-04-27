"""
UMLS REST API Client
Rate-limited, with retry logic and comprehensive error handling.

Base URL: https://uts-ws.nlm.nih.gov/rest
Auth: API key as query parameter (?apiKey=...)
Ref: https://documentation.uts.nlm.nih.gov/rest/home.html
"""

import time
import logging
import threading
import requests
from typing import Optional

import config

logger = logging.getLogger(__name__)


class UMLSClient:
    """Thread-safe UMLS REST API client with shared rate limiting.

    The rate limiter is global across threads: regardless of how many workers
    call _get concurrently, requests are spaced by at least rate_limit_sleep
    seconds. The HTTP request itself runs without holding the lock so multiple
    requests can be in flight simultaneously, which is what allows parallelism
    to actually saturate the UMLS 20 req/s ceiling.
    """

    def __init__(
        self,
        api_key: str = None,
        version: str = None,
        rate_limit_sleep: float = None,
    ):
        self.api_key = api_key or config.UMLS_API_KEY
        self.version = version or config.UMLS_VERSION
        self.base_url = config.UMLS_BASE_URL
        self.rate_limit_sleep = rate_limit_sleep or config.UMLS_RATE_LIMIT_SLEEP

        self._request_count = 0
        self._rate_lock = threading.Lock()
        self._last_request_t = 0.0
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    @property
    def request_count(self) -> int:
        return self._request_count

    def _wait_for_rate_limit(self):
        """Block until the next request slot is available, then claim it."""
        with self._rate_lock:
            now = time.monotonic()
            wait = self._last_request_t + self.rate_limit_sleep - now
            if wait > 0:
                time.sleep(wait)
                self._last_request_t = now + wait
            else:
                self._last_request_t = now
            self._request_count += 1

    def _get(
        self,
        endpoint: str,
        params: dict = None,
        retries: int = 2,
    ) -> Optional[dict]:
        """
        Rate-limited GET request with retry logic.

        Returns parsed JSON dict, or None on failure / 404.
        """
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        url = f"{self.base_url}{endpoint}"

        for attempt in range(retries + 1):
            self._wait_for_rate_limit()

            try:
                r = self._session.get(url, params=params, timeout=30)

                if r.status_code == 404:
                    return None

                if r.status_code == 429:
                    # Rate limited — back off and retry
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited (429), waiting {wait}s...")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                return r.json()

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Timeout for {endpoint} (attempt {attempt + 1}/{retries + 1})"
                )
                if attempt < retries:
                    time.sleep(1)
                    continue
                return None

            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP error for {endpoint}: {e}")
                return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {endpoint}: {e}")
                return None

            except ValueError:
                logger.warning(f"Invalid JSON from {endpoint}")
                return None

        return None

    # ──────────────────────────────────────────────────────────────────
    # Search API
    # ──────────────────────────────────────────────────────────────────

    def search(
        self,
        term: str,
        search_type: str = "exact",
        page_size: int = 200,
        sabs: str = None,
    ) -> list[dict]:
        """
        Search UMLS for a term.

        Args:
            term: search string
            search_type: "exact", "normalizedString", "normalizedWords", "words"
            page_size: max results per page (up to 200)
            sabs: restrict to specific source vocabularies (e.g. "SNOMEDCT_US")

        Returns: list of result dicts with keys: ui, name, rootSource, uri
        """
        params = {
            "string": term,
            "searchType": search_type,
            "pageSize": min(page_size, 200),
        }
        if sabs:
            params["sabs"] = sabs

        data = self._get(f"/search/{self.version}", params)

        if not data:
            return []

        result = data.get("result", {})
        results_list = result.get("results", [])

        # Filter out the "NO RESULTS" sentinel
        return [r for r in results_list if r.get("ui", "NONE") != "NONE"]

    def search_exact(self, term: str, sabs: str = None) -> list[dict]:
        """Exact match search."""
        return self.search(
            term, search_type="exact",
            page_size=config.MAX_SEARCH_RESULTS_EXACT, sabs=sabs,
        )

    def search_normalized(self, term: str) -> list[dict]:
        """NormalizedString search (absorbs case/plural differences)."""
        return self.search(
            term, search_type="normalizedString",
            page_size=config.MAX_SEARCH_RESULTS_NORMALIZED,
        )

    def search_words(self, term: str) -> list[dict]:
        """Words search (all words present, loosest matching)."""
        return self.search(
            term, search_type="words",
            page_size=config.MAX_SEARCH_RESULTS_WORDS,
        )

    # ──────────────────────────────────────────────────────────────────
    # Concept API
    # ──────────────────────────────────────────────────────────────────

    def get_concept(self, cui: str) -> Optional[dict]:
        """
        Retrieve concept info for a CUI.

        Returns dict with: name, semanticTypes, atomCount, relationCount, etc.
        """
        data = self._get(f"/content/{self.version}/CUI/{cui}")
        if data and "result" in data:
            return data["result"]
        return None

    # ──────────────────────────────────────────────────────────────────
    # Relations API (core of 1-hop subgraph)
    # ──────────────────────────────────────────────────────────────────

    def get_relations(
        self,
        cui: str,
        page_size: int = None,
        max_pages: int = 10,
    ) -> list[dict]:
        """
        Retrieve ALL relations for a CUI (paginated).

        Returns list of relation dicts with:
            relationLabel, additionalRelationLabel, relatedIdName,
            relatedId, rootSource, etc.
        """
        page_size = page_size or config.MAX_RELATIONS_PAGE_SIZE
        all_relations = []
        page = 1

        while page <= max_pages:
            params = {"pageSize": page_size, "pageNumber": page}
            data = self._get(
                f"/content/{self.version}/CUI/{cui}/relations", params
            )

            if not data or "result" not in data:
                break

            results = data["result"]
            if not results:
                break

            all_relations.extend(results)

            # End of results
            if len(results) < page_size:
                break

            page += 1

        return all_relations
