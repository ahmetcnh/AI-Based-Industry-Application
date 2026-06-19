import asyncio
import json
from collections import OrderedDict
from typing import AsyncIterator, Dict, Any, List, Optional
import httpx
import structlog

logger = structlog.get_logger(__name__)

# keep_alive=-1 pins models in Ollama's memory so consecutive requests
# don't pay a multi-second model reload on VRAM-constrained machines.
_OLLAMA_KEEP_ALIVE = -1
# Default Ollama context (2048) silently truncates RAG prompts
# (system role + k chunks + question); 8192 fits them comfortably.
_OLLAMA_NUM_CTX = 8192

_EMBED_CACHE_MAX = 512


async def _ollama_generate(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list,
    stream: bool,
    timeout: int,
) -> AsyncIterator[str]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": _OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0.7, "num_predict": 512, "num_ctx": _OLLAMA_NUM_CTX},
    }
    if stream:
        async with client.stream(
            "POST", f"{base_url}/api/chat", json=payload, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
    else:
        resp = await client.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        yield resp.json().get("message", {}).get("content", "")


DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
# Gemini exposes an OpenAI-compatible endpoint; same payload shape as DeepSeek.
GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


async def _openai_compat_generate(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    model: str,
    messages: list,
    stream: bool,
    timeout: int,
) -> AsyncIterator[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": stream}
    if stream:
        async with client.stream(
            "POST", url,
            json=payload, headers=headers, timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        data = json.loads(line[6:])
                        chunk = data["choices"][0]["delta"].get("content", "")
                        if chunk:
                            yield chunk
                    except (json.JSONDecodeError, KeyError):
                        continue
    else:
        # One quick retry on 429, then give up so the caller can fall back
        # to the local Ollama model instead of stalling for 60s+.
        for attempt in range(2):
            resp = await client.post(
                url,
                json=payload, headers=headers, timeout=timeout,
            )
            if resp.status_code == 429 and attempt < 1:
                logger.warning("API rate limited, retrying once", wait_s=5)
                await asyncio.sleep(5)
                continue
            resp.raise_for_status()
            yield resp.json()["choices"][0]["message"]["content"]
            return


class LLMProvider:
    def __init__(self, settings):
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT)
        # LRU cache: (model, text) -> embedding. The router and the RAG
        # retriever both embed the same question; this makes the second
        # call (and repeated questions) free.
        self._embed_cache: "OrderedDict[tuple, List[float]]" = OrderedDict()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _parse_model_id(self, model_id: Optional[str]) -> tuple:
        if model_id == "default":
            model_id = None
        model_id = model_id or self.settings.DEFAULT_MODEL \
            or f"ollama:{self.settings.OLLAMA_GENERATION_MODEL}"
        if model_id.startswith("ollama:"):
            return "ollama", model_id[7:]
        if model_id in self.settings.deepseek_models_list:
            return "deepseek", model_id
        if model_id.startswith("deepseek"):
            return "deepseek", model_id
        if model_id in self.settings.gemini_models_list:
            return "gemini", model_id
        if model_id.startswith("gemini"):
            return "gemini", model_id
        return "ollama", model_id

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        skip_ollama_fallback: bool = False,
    ) -> str:
        provider, model_name = self._parse_model_id(model)
        messages = self._build_messages(prompt, system_prompt)
        result = ""
        if provider == "deepseek" and self.settings.DEEPSEEK_API_KEY:
            async for chunk in _openai_compat_generate(
                self._client, DEEPSEEK_CHAT_URL, self.settings.DEEPSEEK_API_KEY,
                model_name, messages, False, self.settings.OLLAMA_TIMEOUT
            ):
                result += chunk
        elif provider == "gemini" and self.settings.GEMINI_API_KEY:
            try:
                async for chunk in _openai_compat_generate(
                    self._client, GEMINI_CHAT_URL, self.settings.GEMINI_API_KEY,
                    model_name, messages, False, self.settings.OLLAMA_TIMEOUT
                ):
                    result += chunk
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 429:
                    raise
                if skip_ollama_fallback:
                    raise
                logger.warning("gemini_429_fallback_ollama", model=model_name)
                result = ""
                async for chunk in _ollama_generate(
                    self._client, self.settings.OLLAMA_BASE_URL,
                    self.settings.OLLAMA_GENERATION_MODEL, messages,
                    False, self.settings.OLLAMA_TIMEOUT
                ):
                    result += chunk
        else:
            async for chunk in _ollama_generate(
                self._client, self.settings.OLLAMA_BASE_URL, model_name, messages,
                False, self.settings.OLLAMA_TIMEOUT
            ):
                result += chunk
        return result

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        skip_ollama_fallback: bool = False,
    ) -> AsyncIterator[str]:
        provider, model_name = self._parse_model_id(model)
        messages = self._build_messages(prompt, system_prompt)
        if provider == "deepseek" and self.settings.DEEPSEEK_API_KEY:
            async for chunk in _openai_compat_generate(
                self._client, DEEPSEEK_CHAT_URL, self.settings.DEEPSEEK_API_KEY,
                model_name, messages, True, self.settings.OLLAMA_TIMEOUT
            ):
                yield chunk
        elif provider == "gemini" and self.settings.GEMINI_API_KEY:
            emitted = False
            try:
                async for chunk in _openai_compat_generate(
                    self._client, GEMINI_CHAT_URL, self.settings.GEMINI_API_KEY,
                    model_name, messages, True, self.settings.OLLAMA_TIMEOUT
                ):
                    emitted = True
                    yield chunk
            except httpx.HTTPStatusError as e:
                # 429 raises before any chunk is streamed; if nothing was
                # emitted yet we can transparently restart on local Ollama.
                if e.response.status_code != 429 or emitted:
                    raise
                if skip_ollama_fallback:
                    raise
                logger.warning("gemini_429_fallback_ollama", model=model_name)
                async for chunk in _ollama_generate(
                    self._client, self.settings.OLLAMA_BASE_URL,
                    self.settings.OLLAMA_GENERATION_MODEL, messages,
                    True, self.settings.OLLAMA_TIMEOUT
                ):
                    yield chunk
        else:
            async for chunk in _ollama_generate(
                self._client, self.settings.OLLAMA_BASE_URL, model_name, messages,
                True, self.settings.OLLAMA_TIMEOUT
            ):
                yield chunk

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts using /api/embed (Ollama >=0.2). Falls back to sequential get_embedding."""
        if not texts:
            return []
        # Check LRU cache first; collect uncached texts
        model = self.settings.OLLAMA_EMBEDDING_MODEL
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        for i, text in enumerate(texts):
            key = (model, text)
            cached = self._embed_cache.get(key)
            if cached is not None:
                self._embed_cache.move_to_end(key)
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            batch_size = 32
            fetched: List[List[float]] = []
            try:
                for start in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[start:start + batch_size]
                    resp = await self._client.post(
                        f"{self.settings.OLLAMA_BASE_URL}/api/embed",
                        json={"model": model, "input": batch, "keep_alive": _OLLAMA_KEEP_ALIVE},
                        timeout=120,
                    )
                    if resp.status_code == 404:
                        raise ValueError("api/embed not available")
                    resp.raise_for_status()
                    fetched.extend(resp.json()["embeddings"])
            except Exception:
                # Fallback to sequential get_embedding
                for text in uncached_texts:
                    fetched.append(await self.get_embedding(text))

            for idx, emb in zip(uncached_indices, fetched):
                key = (model, texts[idx])
                self._embed_cache[key] = emb
                if len(self._embed_cache) > _EMBED_CACHE_MAX:
                    self._embed_cache.popitem(last=False)
                results[idx] = emb

        return [r for r in results if r is not None]

    async def get_embedding(self, text: str) -> List[float]:
        cache_key = (self.settings.OLLAMA_EMBEDDING_MODEL, text)
        cached = self._embed_cache.get(cache_key)
        if cached is not None:
            self._embed_cache.move_to_end(cache_key)
            return cached

        payload = {
            "model": self.settings.OLLAMA_EMBEDDING_MODEL,
            "prompt": text,
            "keep_alive": _OLLAMA_KEEP_ALIVE,
        }
        resp = await self._client.post(
            f"{self.settings.OLLAMA_BASE_URL}/api/embeddings", json=payload, timeout=60
        )
        resp.raise_for_status()
        embedding = resp.json()["embedding"]

        self._embed_cache[cache_key] = embedding
        if len(self._embed_cache) > _EMBED_CACHE_MAX:
            self._embed_cache.popitem(last=False)
        return embedding

    async def list_ollama_models(self) -> List[str]:
        try:
            resp = await self._client.get(
                f"{self.settings.OLLAMA_BASE_URL}/api/tags", timeout=5
            )
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    async def check_health(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {}
        try:
            models = await self.list_ollama_models()
            health["ollama"] = {"available": True, "models": models}
        except Exception as e:
            health["ollama"] = {"available": False, "error": str(e)}
        if self.settings.DEEPSEEK_API_KEY:
            health["deepseek"] = {"available": True,
                                   "models": self.settings.deepseek_models_list}
        return health
