"""
Full-system smoke test for the Coiltech Industrial Automation Platform.

This script is intentionally deterministic: it uses a fake LLM and fake RAG
store so it can validate the application without requiring Ollama, OpenAI, or
Anthropic to be running.

Run:
    python full_system_test.py
    python full_system_test.py --skip-pytest
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from agents.digital_twin_agent import DigitalTwinAgent
from agents.orchestrator import Orchestrator
from agents.router_agent import RouterAgent
from config.settings import RAG_COLLECTION_REGISTRY
from database.db_manager import SQLiteManager
from demo.seed_demo_data import reset_database, seed_relational_data
from rag.rag_pipeline import RAGPipeline


ROUTING_CASES = [
    ("What compact drive models do you offer?", "product_lookup", "ProductCatalogAgent"),
    ("We need 2.5 mm steel at 600 mm width. Which model fits?", "application_engineering", "ApplicationEngineeringAgent"),
    ("Configure a feed line BOM with decoiler and servo feeder", "configuration_request", "ConfigurationAgent"),
    ("Send me the quotation for Arcelik compact line", "quotation_request", "QuotationAgent"),
    ("What is the status of project PRJ-2025-014?", "project_status", "ProjectExecutionAgent"),
    ("Show the commissioning checklist for CTL line startup", "commissioning_support", "InstallationCommissioningAgent"),
    ("Open service tickets for Arcelik compact line", "maintenance_request", "ServiceMaintenanceAgent"),
    ("Do we have SM-400 in stock?", "spare_parts_lookup", "SparePartsAgent"),
    ("List the installed base at Tofas", "installed_base_query", "ServiceMaintenanceAgent"),
    ("Any encoder drift alarms right now?", "anomaly_detection", "DigitalTwinAgent"),
    ("Where is the CE document for the feed line?", "quality_compliance", "QualityComplianceAgent"),
    ("Give me an executive summary of pipeline and service risk", "executive_summary", "ManagementReportingAgent"),
    ("Hello, what can this platform do?", "general_knowledge", "GeneralAgent"),
]


class CheckRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: List[str] = []

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def check(self, name: str, func: Callable[[], Any]) -> Any:
        try:
            result = func()
            self.passed += 1
            print(f"[PASS] {name}")
            return result
        except Exception as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            self.failed.append(message)
            print(f"[FAIL] {message}")
            return None

    async def check_async(self, name: str, func: Callable[[], Any]) -> Any:
        try:
            result = await func()
            self.passed += 1
            print(f"[PASS] {name}")
            return result
        except Exception as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            self.failed.append(message)
            print(f"[FAIL] {message}")
            return None

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def finish(self) -> int:
        print("\n=== Summary ===")
        print(f"Passed: {self.passed}")
        print(f"Failed: {len(self.failed)}")
        if self.failed:
            print("\nFailures:")
            for item in self.failed:
                print(f"- {item}")
            return 1
        print("All full-system checks passed.")
        return 0


class FakeLLMClient:
    def _extract_user_text(self, prompt: str) -> str:
        text = prompt.lower()
        for pattern in (
            r"user message:\s*(.*?)\s*(?:context:|respond with a json object:|json response:|$)",
            r"user query:\s*(.*?)\s*(?:context:|respond with a json object:|json response:|$)",
            r"user question:\s*(.*?)\s*$",
        ):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text

    def _classify(self, prompt: str) -> str:
        text = self._extract_user_text(prompt)
        if "which model" in text or "thickness" in text or "strip width" in text:
            return "application_engineering"
        if "project" in text or "milestone" in text or "delivery" in text:
            return "project_status"
        if "commissioning" in text or "installation" in text or "startup" in text:
            return "commissioning_support"
        if "open service ticket" in text or "service ticket" in text or "maintenance" in text or "repair" in text:
            return "maintenance_request"
        if "installed base" in text or "warranty" in text or "serial number" in text:
            return "installed_base_query"
        if "alarm" in text or "anomaly" in text or "servo load" in text or "encoder drift" in text:
            return "anomaly_detection"
        if "summary" in text or "pipeline" in text or "executive" in text:
            return "executive_summary"
        if "spare" in text or "stock" in text or "part number" in text:
            return "spare_parts_lookup"
        if "configure" in text or "bom" in text or "line setup" in text:
            return "configuration_request"
        if "quote" in text or "quotation" in text or "price" in text or "rfq" in text:
            return "quotation_request"
        if re.search(r"\bce\b", text) or "quality" in text or "compliance" in text:
            return "quality_compliance"
        if "compact drive" in text or "model" in text or "catalog" in text:
            return "product_lookup"
        return "general_knowledge"

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: str = "routing",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        category = self._classify(prompt)
        return {
            "category": category,
            "confidence": 0.91,
            "reasoning": f"fake classifier selected {category}",
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: str = "generation",
        **kwargs: Any,
    ) -> str:
        question = self._extract_user_text(prompt)
        if "database schema:" in prompt.lower() and "user question:" in prompt.lower():
            if "quotation" in question or "quote" in question:
                return "SELECT * FROM quotations LIMIT 5"
            if "project" in question:
                return "SELECT * FROM v_project_status LIMIT 5"
            if "service" in question or "maintenance" in question:
                return "SELECT * FROM v_open_service_tickets LIMIT 5"
            if "spare" in question or "stock" in question or "stok" in question or "sm-400" in question:
                return "SELECT * FROM v_spare_parts_low_stock LIMIT 5"
            if "alert" in question or "anomaly" in question or "alarm" in question:
                return "SELECT * FROM v_active_alerts LIMIT 5"
            if "installed" in question or "warranty" in question:
                return "SELECT * FROM v_installed_base_overview LIMIT 5"
            if "quality" in question or "ce" in question:
                return "SELECT * FROM quality_events LIMIT 5"
            return "SELECT * FROM product_families LIMIT 5"
        return "Stubbed response for Coiltech domain."

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: str = "generation",
        **kwargs: Any,
    ):
        for token in ["Stubbed ", "stream ", "response"]:
            yield token

    async def check_health(self) -> Dict[str, Any]:
        return {
            "fake": {
                "available": True,
                "routing_model": "fake-routing",
                "generation_model": "fake-generation",
            }
        }


class FakeVectorStore:
    def similarity_search(
        self,
        query: str,
        department: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "id": f"fake-{department}-1",
                "content": f"Fake retrieved context for {department}: {query}",
                "score": 0.92,
                "metadata": {
                    "source": f"{department}.txt",
                    "title": f"{department} smoke-test source",
                },
            }
        ][:k]


class FakeRAGPipeline:
    def __init__(self, resolver: RAGPipeline) -> None:
        self.vector_store = FakeVectorStore()
        self._resolver = resolver
        self.ingested_files: List[Dict[str, Any]] = []

    def resolve_collection(
        self,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
    ) -> str:
        return self._resolver.resolve_collection(collection_name, metadata, file_path)

    async def ingest_async(
        self,
        file_path: str,
        department: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        collection = self.resolve_collection(department, metadata, file_path)
        self.ingested_files.append(
            {"file_path": file_path, "collection": collection, "metadata": metadata or {}}
        )
        return {
            "file": file_path,
            "department": collection,
            "collection": collection,
            "documents_loaded": 1,
            "chunks_created": 1,
            "chunks_stored": 1,
            "collection_total": len(self.ingested_files),
        }

    def list_all_collections(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": f"coiltech_{key}",
                "collection_key": key,
                "count": 1 if key in {"products_compact_drives", "service_maintenance"} else 0,
                "metadata": {"fake": True, **meta},
            }
            for key, meta in RAG_COLLECTION_REGISTRY.items()
        ]


class StatelessSQLiteManager(SQLiteManager):
    """SQLite manager variant for TestClient thread boundaries."""

    def _execute_once(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
    ) -> Any:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or ())
            if query.strip().upper().startswith("SELECT"):
                if fetch_one:
                    row = cursor.fetchone()
                    return dict(row) if row else None
                return [dict(row) for row in cursor.fetchall()]
            conn.commit()
            return {"affected_rows": cursor.rowcount, "lastrowid": cursor.lastrowid}
        finally:
            cursor.close()
            conn.close()

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
    ) -> Any:
        return self._execute_once(query, params, fetch_one)

    async def async_execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
    ) -> Any:
        return await asyncio.to_thread(self._execute_once, query, params, fetch_one)

    async def register_uploaded_document(
        self,
        original_filename: str,
        file_path: str,
        rag_collection: str,
        stored_filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        document_role: str = "general_document",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata_json = json.dumps(metadata or {}, ensure_ascii=True)

        def insert() -> Dict[str, Any]:
            result = self._execute_once(
                """
                INSERT INTO uploaded_documents (
                    original_filename, stored_filename, file_path, mime_type, file_size_bytes,
                    document_role, rag_collection, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    original_filename,
                    stored_filename,
                    file_path,
                    mime_type,
                    file_size_bytes,
                    document_role,
                    rag_collection,
                    metadata_json,
                ),
            )
            return {"affected_rows": result["affected_rows"], "id": result["lastrowid"]}

        return await asyncio.to_thread(insert)

    def close(self) -> None:
        self._connection = None

    async def async_close(self) -> None:
        self._async_connection = None


def make_test_system(tmpdir: Path) -> tuple[SQLiteManager, FakeLLMClient, FakeRAGPipeline, Orchestrator]:
    db = SQLiteManager(str(tmpdir / "coiltech_full_system_test.db"))
    db.initialize_schema()
    reset_database(db)
    seed_relational_data(db)

    fake_llm = FakeLLMClient()
    resolver = RAGPipeline(vector_store=FakeVectorStore(), llm_client=fake_llm)
    fake_rag = FakeRAGPipeline(resolver)
    orchestrator = Orchestrator(llm_client=fake_llm, db_manager=db, rag_pipeline=fake_rag)
    return db, fake_llm, fake_rag, orchestrator


async def run_database_checks(runner: CheckRunner, db: SQLiteManager, fake_llm: FakeLLMClient) -> None:
    runner.section("Database and Seed Data")

    expected_counts = {
        "customers": 6,
        "product_families": 20,
        "product_models": 12,
        "projects": 4,
        "quotations": 3,
        "installed_base": 5,
        "service_tickets": 4,
        "spare_parts": 6,
        "machine_alerts": 4,
        "technical_documents": 7,
    }

    for table, minimum in expected_counts.items():
        runner.check(
            f"{table} has at least {minimum} rows",
            lambda table=table, minimum=minimum: runner.require(
                db.execute_query(f"SELECT COUNT(*) AS count FROM {table}", fetch_one=True)["count"] >= minimum,
                f"{table} count is below {minimum}",
            ),
        )

    await runner.check_async(
        "product families helper",
        lambda: _assert_len(awaitable=db.get_product_families(), minimum=20),
    )
    await runner.check_async(
        "product models helper",
        lambda: _assert_len(awaitable=db.get_product_models(), minimum=12),
    )
    await runner.check_async("project detail helper", lambda: _check_project_detail(db))
    await runner.check_async("quotation detail helper", lambda: _check_quotation_detail(db))
    await runner.check_async("service ticket helper", lambda: _assert_len(awaitable=db.get_service_tickets(), minimum=4))
    await runner.check_async("spare parts low-stock helper", lambda: _check_low_stock(db))
    await runner.check_async("active alerts helper", lambda: _assert_len(awaitable=db.get_active_alerts(), minimum=4))
    await runner.check_async("pipeline summary helper", lambda: _assert_len(awaitable=db.get_pipeline_summary(), minimum=1))
    await runner.check_async("safe NL-to-SQL succeeds", lambda: _check_nl_to_sql(db, fake_llm))
    runner.check(
        "unsafe SQL is blocked",
        lambda: runner.require(
            db._validate_safe_select("DELETE FROM customers")[0] is False,
            "DELETE statement was not blocked",
        ),
    )


async def _assert_len(awaitable: Any, minimum: int) -> None:
    rows = await awaitable
    if len(rows) < minimum:
        raise AssertionError(f"expected at least {minimum} rows, got {len(rows)}")


async def _check_project_detail(db: SQLiteManager) -> None:
    projects = await db.get_projects()
    target = next((item for item in projects if item["code"] == "PRJ-2025-014"), None)
    if not target:
        raise AssertionError("PRJ-2025-014 not found")
    detail = await db.get_project_detail(target["id"])
    if not detail or detail["customer_name"] != "Tofas":
        raise AssertionError("project detail did not include expected Tofas project")
    if not detail["milestones"] or not detail["machines"]:
        raise AssertionError("project detail missing milestones or machines")


async def _check_quotation_detail(db: SQLiteManager) -> None:
    quotations = await db.get_quotations()
    target = next((item for item in quotations if item["code"] == "QUO-2026-001"), None)
    if not target:
        raise AssertionError("QUO-2026-001 not found")
    detail = await db.get_quotation_detail(target["id"])
    if not detail or detail["customer_name"] != "Arcelik":
        raise AssertionError("quotation detail did not include expected Arcelik quote")
    if not any(item["model_code"] == "CD-300" for item in detail["items"]):
        raise AssertionError("quotation detail missing CD-300 item")


async def _check_low_stock(db: SQLiteManager) -> None:
    rows = await db.get_spare_parts_low_stock()
    if not any(row["part_number"] == "SM-400" for row in rows):
        raise AssertionError("SM-400 was not found in low-stock view")


async def _check_nl_to_sql(db: SQLiteManager, fake_llm: FakeLLMClient) -> None:
    result = await db.nl_to_sql("Show all quotations", fake_llm)
    if not result.get("success"):
        raise AssertionError(result.get("error") or "NL-to-SQL failed")
    if not result.get("generated_sql", "").upper().startswith("SELECT"):
        raise AssertionError("NL-to-SQL did not generate SELECT")


async def run_routing_checks(runner: CheckRunner, fake_llm: FakeLLMClient) -> None:
    runner.section("Router")
    router = RouterAgent(llm_client=fake_llm)
    for prompt, expected_intent, expected_agent in ROUTING_CASES:
        async def check_case(
            prompt: str = prompt,
            expected_intent: str = expected_intent,
            expected_agent: str = expected_agent,
        ) -> None:
            result = await router.classify_intent(prompt)
            if result.category != expected_intent:
                raise AssertionError(f"expected {expected_intent}, got {result.category}")
            agent = router.get_agent_for_intent(result)
            if agent != expected_agent:
                raise AssertionError(f"expected {expected_agent}, got {agent}")

        await runner.check_async(f"{expected_intent} -> {expected_agent}", check_case)

    async def check_stock_code_fast_path() -> None:
        result = await router.classify_intent("SM-400 var mı?")
        if result.category != "spare_parts_lookup":
            raise AssertionError(f"expected spare_parts_lookup, got {result.category}")
        if not (result.metadata or {}).get("fast_path"):
            raise AssertionError("stock code query did not use router fast path")
        agent = router.get_agent_for_intent(result)
        if agent != "SparePartsAgent":
            raise AssertionError(f"expected SparePartsAgent, got {agent}")

    await runner.check_async("stock code fast path -> SparePartsAgent", check_stock_code_fast_path)


async def run_orchestrator_checks(runner: CheckRunner, orchestrator: Orchestrator) -> None:
    runner.section("Orchestrator and Agents")
    for prompt, expected_intent, expected_agent in ROUTING_CASES:
        async def check_case(
            prompt: str = prompt,
            expected_intent: str = expected_intent,
            expected_agent: str = expected_agent,
        ) -> None:
            result = await orchestrator.route_and_execute(prompt, session_id=f"full-{expected_intent}")
            if result.intent["category"] != expected_intent:
                raise AssertionError(f"expected intent {expected_intent}, got {result.intent['category']}")
            if result.agent_used != expected_agent:
                raise AssertionError(f"expected agent {expected_agent}, got {result.agent_used}")
            if not result.content:
                raise AssertionError("empty agent response content")
            if result.confidence < 0:
                raise AssertionError("invalid confidence")
            if expected_intent == "product_lookup":
                matching = (result.data or {}).get("matching_models") or []
                codes = {item.get("model_code") for item in matching}
                if not {"CD-300", "CD-600"}.issubset(codes):
                    raise AssertionError(f"compact drive matching models missing: {sorted(codes)}")

        await runner.check_async(f"{expected_agent} handles {expected_intent}", check_case)


def run_rag_checks(runner: CheckRunner, fake_llm: FakeLLMClient) -> None:
    runner.section("RAG Collection Resolution")
    pipeline = RAGPipeline(vector_store=FakeVectorStore(), llm_client=fake_llm)
    cases = [
        (None, {"title": "Press transfer robot service bulletin"}, "robot_service.txt", "products_press_transfer_robots"),
        (None, {"doc_type": "certificate", "title": "CE Compliance"}, "ce.txt", "quality_documents"),
        (None, {"title": "Quotation terms"}, "quotation.txt", "sales_and_quotations"),
        ("spare_parts", {}, "anything.txt", "spare_parts"),
    ]
    for collection, metadata, file_path, expected in cases:
        runner.check(
            f"{file_path} -> {expected}",
            lambda collection=collection, metadata=metadata, file_path=file_path, expected=expected: runner.require(
                pipeline.resolve_collection(collection, metadata, file_path) == expected,
                "wrong RAG collection",
            ),
        )


def run_digital_twin_checks(runner: CheckRunner, db: SQLiteManager, fake_llm: FakeLLMClient, fake_rag: FakeRAGPipeline) -> None:
    runner.section("Digital Twin Logic")
    agent = DigitalTwinAgent(llm_client=fake_llm, db_manager=db, rag_pipeline=fake_rag)

    def check_anomalies() -> None:
        anomalies = agent._detect_anomalies(
            [
                {
                    "machine_code": "IB-0003",
                    "machine_name": "Arcelik Compact Feed Line",
                    "timestamp": "2026-04-16 08:12:00",
                    "servo_load_pct": 89.0,
                    "feed_length_actual_mm": 300.28,
                    "feed_length_target_mm": 300.0,
                    "bearing_temperature_c": 82.0,
                    "vibration_mm_s": 8.1,
                    "air_pressure_bar": 4.2,
                    "lubrication_ok": 1,
                    "encoder_drift_mm": 0.24,
                }
            ]
        )
        types = {item.anomaly_type for item in anomalies}
        expected = {
            "servo_overload",
            "feed_deviation",
            "high_bearing_temp",
            "high_vibration",
            "low_air_pressure",
            "encoder_drift",
        }
        missing = expected - types
        if missing:
            raise AssertionError(f"missing anomaly types: {sorted(missing)}")

    runner.check("threshold anomaly detection", check_anomalies)


def run_api_checks(
    runner: CheckRunner,
    tmpdir: Path,
    db_path: Path,
    fake_llm: FakeLLMClient,
    fake_rag: FakeRAGPipeline,
) -> None:
    runner.section("FastAPI Surface")

    try:
        from fastapi.testclient import TestClient
        import api.main as api_main
    except Exception as exc:
        runner.failed.append(f"FastAPI import: {type(exc).__name__}: {exc}")
        print(f"[FAIL] FastAPI import: {exc}")
        return

    db_api = StatelessSQLiteManager(str(db_path))
    api_orchestrator = Orchestrator(llm_client=fake_llm, db_manager=db_api, rag_pipeline=fake_rag)

    upload_dir = tmpdir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    api_main.db_manager = db_api
    api_main.rag_pipeline = fake_rag
    api_main.orchestrator = api_orchestrator
    api_main.LLMClient = lambda: fake_llm
    api_main.settings.UPLOAD_DIR = str(upload_dir)

    client = TestClient(api_main.app)

    def expect_status(method: str, url: str, expected: int = 200, **kwargs: Any) -> Dict[str, Any]:
        response = getattr(client, method)(url, **kwargs)
        if response.status_code != expected:
            raise AssertionError(f"{method.upper()} {url} returned {response.status_code}: {response.text[:300]}")
        if response.content:
            return response.json()
        return {}

    runner.check("GET /health", lambda: runner.require(expect_status("get", "/health")["status"] == "ok", "bad health"))
    runner.check("GET /status", lambda: runner.require(expect_status("get", "/status")["status"] in {"healthy", "degraded"}, "bad status"))
    runner.check("GET /agents", lambda: runner.require(expect_status("get", "/agents")["agents"], "no agents returned"))
    runner.check(
        "GET /v1/models",
        lambda: runner.require(
            any(model["id"] == "coiltech-platform" for model in expect_status("get", "/v1/models")["data"]),
            "coiltech-platform model missing",
        ),
    )
    runner.check("GET /collections", lambda: runner.require(len(expect_status("get", "/collections")["collections"]) >= 15, "collections missing"))
    runner.check("GET /product-families", lambda: runner.require(expect_status("get", "/product-families")["count"] >= 20, "families missing"))
    runner.check("GET /projects", lambda: runner.require(expect_status("get", "/projects")["count"] >= 4, "projects missing"))
    runner.check("GET /quotations", lambda: runner.require(expect_status("get", "/quotations")["count"] >= 3, "quotations missing"))
    runner.check("GET /installed-base", lambda: runner.require(expect_status("get", "/installed-base")["count"] >= 5, "installed base missing"))
    runner.check("GET /service-tickets", lambda: runner.require(expect_status("get", "/service-tickets")["count"] >= 4, "tickets missing"))
    runner.check("GET /spare-parts", lambda: runner.require(expect_status("get", "/spare-parts")["count"] >= 6, "spare parts missing"))
    runner.check("GET /alerts", lambda: runner.require(expect_status("get", "/alerts")["count"] >= 4, "alerts missing"))

    def check_chat() -> None:
        data = expect_status(
            "post",
            "/chat",
            json={"message": "Do we have SM-400 in stock?", "session_id": "api-smoke"},
        )
        runner.require(data["agent_used"] == "SparePartsAgent", f"wrong chat agent: {data['agent_used']}")
        runner.require(data["data"]["direct_sql"] is True, "stock query did not use direct SQL fast path")

    runner.check("POST /chat", check_chat)

    def check_chat_turkish_stock() -> None:
        data = expect_status(
            "post",
            "/chat",
            json={"message": "SM-400 var mı", "session_id": "api-stock-tr"},
        )
        runner.require(data["agent_used"] == "SparePartsAgent", f"wrong chat agent: {data['agent_used']}")
        runner.require(data["data"]["direct_sql"] is True, "Turkish stock query did not use direct SQL fast path")
        runner.require(data["data"]["deterministic_stock_lookup"] is True, "Turkish stock query did not use deterministic lookup")
        runner.require("SM-400" in data["content"], f"stock response missing part code: {data['content']}")

    runner.check("POST /chat Turkish stock", check_chat_turkish_stock)

    runner.check(
        "GET /sessions/{session_id}",
        lambda: runner.require(expect_status("get", "/sessions/api-smoke")["session"]["message_count"] >= 2, "session not stored"),
    )
    runner.check(
        "DELETE /sessions/{session_id}",
        lambda: runner.require(expect_status("delete", "/sessions/api-smoke")["status"] == "success", "session not deleted"),
    )

    runner.check(
        "POST /query",
        lambda: runner.require(
            expect_status("post", "/query", params={"question": "Show all quotations"})["status"] == "success",
            "query endpoint failed",
        ),
    )

    def check_openai_compat() -> None:
        data = expect_status(
            "post",
            "/v1/chat/completions",
            json={
                "model": "coiltech-platform",
                "messages": [{"role": "user", "content": "What compact drive models do you offer?"}],
            },
        )
        runner.require(data["choices"][0]["message"]["content"], "empty OpenAI-compatible content")

    runner.check("POST /v1/chat/completions", check_openai_compat)

    def check_openai_compat_escapes_document_hint() -> None:
        wrapped_prompt = """
        Use the following document context:
        Midterm Report.pdf says the project is about air quality monitoring.

        Question: show spare part stocks
        Answer:
        """
        data = expect_status(
            "post",
            "/v1/chat/completions",
            json={
                "model": "coiltech-platform",
                "collection_name": "special_projects",
                "messages": [{"role": "user", "content": wrapped_prompt}],
            },
        )
        content = data["choices"][0]["message"]["content"]
        runner.require("Database query completed" in content, f"OpenAI-compatible stock query did not use DB path: {content}")
        runner.require("v_spare_parts_low_stock" in content, f"wrong SQL path for stock query: {content}")

    runner.check("POST /v1/chat/completions escapes document hint", check_openai_compat_escapes_document_hint)

    def check_excel_analysis() -> None:
        import pandas as pd

        buffer = io.BytesIO()
        df = pd.DataFrame(
            {
                "Customer": ["Arcelik", "Arcelik", "Tofas"],
                "Amount": [1000, 1500, 2000],
                "Quantity": [2, 3, 4],
            }
        )
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sales", index=False)
        payload = buffer.getvalue()

        data = expect_status(
            "post",
            "/analyze-excel",
            files={
                "file": (
                    "sales_smoke.xlsx",
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            data={
                "operation": "sum",
                "sheet_name": "Sales",
                "value_column": "Amount",
                "group_by": "Customer",
            },
        )
        grouped = data["result"]["sheets"]["Sales"]["grouped"]
        totals = {row["Customer"]: row["Amount"] for row in grouped}
        runner.require(totals["Arcelik"] == 2500, f"wrong Arcelik total: {totals}")
        runner.require(totals["Tofas"] == 2000, f"wrong Tofas total: {totals}")

    runner.check("POST /analyze-excel", check_excel_analysis)

    def check_upload() -> None:
        data = expect_status(
            "post",
            "/upload",
            files={"file": ("smoke_test_doc.txt", b"Compact drive smoke-test document", "text/plain")},
            data={"collection_name": "products_compact_drives", "title": "Smoke Test Doc", "doc_type": "manual"},
        )
        runner.require(data["status"] == "success", "upload failed")
        runner.require(data["chunks_stored"] == 1, "fake RAG ingestion did not store one chunk")

    runner.check("POST /upload", check_upload)
    client.close()
    db_api.close()


def run_pytest_checks(runner: CheckRunner) -> None:
    runner.section("Existing Pytest Suite")

    def execute() -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            if result.stderr.strip():
                print(result.stderr.strip())
            raise AssertionError(f"pytest returned {result.returncode}")

    runner.check("python -m pytest -q", execute)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run full-system smoke tests.")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip the existing pytest suite.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary test directory.")
    args = parser.parse_args()

    runner = CheckRunner()
    temp_ctx = tempfile.TemporaryDirectory(prefix="coiltech_full_system_")
    tmpdir = Path(temp_ctx.name)

    try:
        runner.section("Setup")
        db, fake_llm, fake_rag, orchestrator = runner.check(
            "initialize isolated seeded system",
            lambda: make_test_system(tmpdir),
        )
        if not db:
            return runner.finish()

        db_path = Path(db.db_path)
        await run_database_checks(runner, db, fake_llm)
        await run_routing_checks(runner, fake_llm)
        run_rag_checks(runner, fake_llm)
        run_digital_twin_checks(runner, db, fake_llm, fake_rag)
        await run_orchestrator_checks(runner, orchestrator)

        await db.async_close()
        db.close()

        run_api_checks(runner, tmpdir, db_path, fake_llm, fake_rag)

        if not args.skip_pytest:
            run_pytest_checks(runner)

        return runner.finish()
    finally:
        if args.keep_temp:
            print(f"\nTemporary test directory kept at: {tmpdir}")
        else:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
