"""
Unit tests for the live backend (no Postgres/Ollama required).
Covers helpers in main.py, pipeline.py, router_agent.py, llm_providers.py,
and answer_cache.py.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# main.py helpers
# ---------------------------------------------------------------------------

from backend.main import (
    _extract_user_message,
    _strip_openwebui_wrapper,
    _openwebui_utility,
    _condense_question,
)
from backend.settings import get_settings


class TestStripOpenWebUIWrapper:
    def test_strips_question_tag(self):
        assert _strip_openwebui_wrapper("<question>What is X?</question>") == "What is X?"

    def test_strips_query_tag(self):
        assert _strip_openwebui_wrapper("<query>Hello</query>") == "Hello"

    def test_no_tag_returns_original(self):
        assert _strip_openwebui_wrapper("Plain text") == "Plain text"

    def test_strips_question_colon_prefix(self):
        result = _strip_openwebui_wrapper("question: What is the issue?")
        assert result == "What is the issue?"

    def test_empty_string(self):
        assert _strip_openwebui_wrapper("") == ""


class TestOpenWebUIUtility:
    def test_title_generation(self):
        msg = "### Task: Generate a concise title for this chat"
        result = _openwebui_utility(msg)
        assert result is not None
        assert len(result) > 0

    def test_tag_generation(self):
        msg = "### Task: Generate 1-3 broad tags categorizing this"
        result = _openwebui_utility(msg)
        assert result is not None
        data = json.loads(result)
        assert "tags" in data

    def test_follow_up_questions(self):
        msg = '### Task: Generate follow-up questions {"follow_ups": []}'
        result = _openwebui_utility(msg)
        assert result is not None

    def test_normal_message_returns_none(self):
        assert _openwebui_utility("What is the issue with my compact drive?") is None

    def test_empty_message_returns_none(self):
        assert _openwebui_utility("") is None


class TestExtractUserMessage:
    def test_simple_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        assert _extract_user_message(msgs) == "Hello"

    def test_picks_last_user_message(self):
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second"},
        ]
        assert _extract_user_message(msgs) == "Second"

    def test_skips_system_role(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Real question"},
        ]
        assert _extract_user_message(msgs) == "Real question"

    def test_content_list(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "List content"}]}]
        assert _extract_user_message(msgs) == "List content"

    def test_strips_openwebui_wrapper(self):
        msgs = [{"role": "user", "content": "<question>Stripped</question>"}]
        assert _extract_user_message(msgs) == "Stripped"

    def test_no_user_message_returns_empty(self):
        msgs = [{"role": "assistant", "content": "Only assistant"}]
        assert _extract_user_message(msgs) == ""


class TestCondenseQuestion:
    @pytest.mark.asyncio
    async def test_no_history_returns_none(self):
        """Single user turn → no condensation needed."""
        msgs = [{"role": "user", "content": "What is X?"}]
        llm = MagicMock()
        result = await _condense_question(msgs, "What is X?", llm)
        assert result is None
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_turns_calls_llm(self):
        """Two user turns → LLM should be called."""
        msgs = [
            {"role": "user", "content": "What is a compact drive?"},
            {"role": "assistant", "content": "A compact drive is..."},
            {"role": "user", "content": "Peki bunun yedek parçası nedir?"},
        ]
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="Compact drive yedek parçası nedir?")
        result = await _condense_question(msgs, "Peki bunun yedek parçası nedir?", llm)
        assert result == "Compact drive yedek parçası nedir?"
        llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        """LLM error → graceful fallback returns None."""
        msgs = [
            {"role": "user", "content": "What is A?"},
            {"role": "assistant", "content": "A is..."},
            {"role": "user", "content": "Ve B?"},
        ]
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        result = await _condense_question(msgs, "Ve B?", llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_condensed_returns_none(self):
        """Empty LLM response → returns None."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="   ")
        result = await _condense_question(msgs, "Q2", llm)
        assert result is None


# ---------------------------------------------------------------------------
# pipeline.py — resolve_collection
# ---------------------------------------------------------------------------

from backend.pipeline import RAGPipeline


class TestResolveCollection:
    def setup_method(self):
        settings = get_settings()
        self.pipeline = RAGPipeline(pool=None, llm=None, settings=settings)

    def test_explicit_collection_name(self):
        assert self.pipeline.resolve_collection("spare_parts") == "spare_parts"

    def test_metadata_keyword_compact_drive(self):
        col = self.pipeline.resolve_collection(
            metadata={"title": "compact drive manual"}
        )
        assert col == "products_compact_drives"

    def test_metadata_keyword_decoiler(self):
        col = self.pipeline.resolve_collection(
            metadata={"doc_type": "decoiler setup guide"}
        )
        assert col == "products_decoilers"

    def test_fallback_to_default(self):
        col = self.pipeline.resolve_collection(metadata={"title": "random stuff"})
        assert col == "service_maintenance"

    def test_file_path_keyword(self):
        # stem is "loop drive spec" — keywords check via lower-case
        col = self.pipeline.resolve_collection(
            file_path="C:/docs/loop drive spec.pdf"
        )
        assert col == "products_loop_drives"


# ---------------------------------------------------------------------------
# router_agent.py — cosine scoring and cache
# ---------------------------------------------------------------------------

from backend.router_agent import RouterAgent, _cosine


class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestRouterAgentCache:
    def setup_method(self):
        settings = get_settings()
        self.router = RouterAgent(pool=None, llm=None, settings=settings)

    def test_cache_empty_initially(self):
        assert len(self.router._cache) == 0

    def test_invalidate_clears_cache(self):
        self.router._cache["key"] = ({"name": "Agent"}, time.time() + 300)
        self.router.invalidate_cache()
        assert len(self.router._cache) == 0


# ---------------------------------------------------------------------------
# llm_providers.py — _parse_model_id
# ---------------------------------------------------------------------------

from backend.llm_providers import LLMProvider


class TestParseModelId:
    def setup_method(self):
        self.llm = LLMProvider(get_settings())

    def test_ollama_prefix(self):
        provider, name = self.llm._parse_model_id("ollama:gemma3")
        assert provider == "ollama"
        assert name == "gemma3"

    def test_default_model_is_ollama(self):
        provider, name = self.llm._parse_model_id(None)
        assert provider == "ollama"

    def test_plain_model_name_is_ollama(self):
        provider, name = self.llm._parse_model_id("llama3")
        assert provider == "ollama"
        assert name == "llama3"

    def test_deepseek_prefix(self):
        provider, name = self.llm._parse_model_id("deepseek-chat")
        assert provider == "deepseek"


# ---------------------------------------------------------------------------
# answer_cache.py — TTL logic (unit test without DB)
# ---------------------------------------------------------------------------

from backend.answer_cache import cache_lookup


class TestAnswerCacheTTL:
    @pytest.mark.asyncio
    async def test_cache_lookup_returns_none_on_db_error(self):
        """If DB raises, cache_lookup returns None gracefully."""
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=MagicMock(
            __aenter__=AsyncMock(side_effect=RuntimeError("no db")),
            __aexit__=AsyncMock(return_value=False),
        ))
        llm = MagicMock()
        llm.get_embedding = AsyncMock(return_value=[0.1] * 768)
        result = await cache_lookup(pool, llm, "test", 1, 1, "col", 86400)
        assert result is None
