"""
Database Manager Module for the Coiltech Industrial Automation Platform.

Provides SQLite database connectivity, query execution, domain-specific
query helpers, and natural language to SQL conversion using LLM.
"""

import os
import sqlite3
import asyncio
import aiosqlite
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from contextlib import asynccontextmanager
import structlog

logger = structlog.get_logger(__name__)


class SQLiteManager:
    """SQLite database manager with async support and NL-to-SQL capabilities."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from config.settings import get_settings
            settings = get_settings()
            db_path = settings.SQLITE_DB_PATH

        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._async_connection: Optional[aiosqlite.Connection] = None
        self._schema_cache: Optional[str] = None

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            logger.info("Database connected", path=self.db_path)
        return self._connection

    async def async_connect(self) -> aiosqlite.Connection:
        if self._async_connection is None:
            self._async_connection = await aiosqlite.connect(self.db_path)
            self._async_connection.row_factory = aiosqlite.Row
            await self._async_connection.execute("PRAGMA foreign_keys = ON")
            logger.info("Async database connected", path=self.db_path)
        return self._async_connection

    @asynccontextmanager
    async def get_connection(self):
        conn = await self.async_connect()
        try:
            yield conn
        finally:
            pass

    # ------------------------------------------------------------------
    # Generic query execution
    # ------------------------------------------------------------------
    def execute_query(
        self, query: str, params: Optional[Tuple] = None, fetch_one: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if query.strip().upper().startswith("SELECT"):
                if fetch_one:
                    row = cursor.fetchone()
                    return dict(row) if row else None
                else:
                    return [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                return {"affected_rows": cursor.rowcount}
        except sqlite3.Error as e:
            logger.error("Query execution failed", query=query, error=str(e))
            raise
        finally:
            cursor.close()

    async def async_execute_query(
        self, query: str, params: Optional[Tuple] = None, fetch_one: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
        async with self.get_connection() as conn:
            try:
                cursor = await conn.execute(query, params) if params else await conn.execute(query)
                if query.strip().upper().startswith("SELECT"):
                    if fetch_one:
                        row = await cursor.fetchone()
                        return dict(row) if row else None
                    else:
                        return [dict(row) for row in await cursor.fetchall()]
                else:
                    await conn.commit()
                    return {"affected_rows": cursor.rowcount}
            except aiosqlite.Error as e:
                logger.error("Async query failed", query=query, error=str(e))
                raise

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------
    def get_table_summary(self) -> str:
        if self._schema_cache:
            return self._schema_cache

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        schema_parts = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            col_info = []
            for col in columns:
                col_name, col_type = col[1], col[2]
                flags = []
                if col[5]:
                    flags.append("PK")
                if col[3]:
                    flags.append("NOT NULL")
                if col[4]:
                    flags.append(f"DEFAULT {col[4]}")
                col_info.append(f"    {col_name} {col_type} {' '.join(flags)}".rstrip())
            schema_parts.append(f"\nTABLE: {table} ({row_count} rows)\n" + "\n".join(col_info))

        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
        views = [row[0] for row in cursor.fetchall()]
        if views:
            schema_parts.append(f"\nVIEWS: {', '.join(views)}")

        self._schema_cache = "\n".join(schema_parts)
        cursor.close()
        conn.close()
        return self._schema_cache

    async def async_get_table_summary(self) -> str:
        if self._schema_cache:
            return self._schema_cache
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_table_summary)

    # ------------------------------------------------------------------
    # NL-to-SQL
    # ------------------------------------------------------------------

    # Tables that NL-to-SQL is ALLOWED to query (read-only, non-sensitive)
    NL_SQL_ALLOWED_TABLES = {
        "customers", "contacts", "product_families", "product_models",
        "machine_capabilities", "project_opportunities", "quotations",
        "quotation_items", "projects", "project_milestones", "project_machines",
        "line_configurations", "bom_components", "installed_base",
        "commissioning_reports", "service_tickets", "maintenance_plans",
        "maintenance_logs", "spare_parts", "spare_part_stock",
        "technical_documents", "uploaded_documents", "sensor_streams", "machine_alerts",
        "line_runs", "quality_events",
        # views
        "v_pipeline_summary", "v_project_status", "v_open_service_tickets",
        "v_spare_parts_low_stock", "v_active_alerts", "v_installed_base_overview",
    }

    def _extract_referenced_tables(self, sql: str) -> List[str]:
        matches = re.findall(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, flags=re.IGNORECASE)
        return [match.strip() for match in matches]

    def _validate_safe_select(self, sql: str) -> Tuple[bool, str]:
        normalized = sql.strip().rstrip(";")
        normalized_upper = normalized.upper()

        if not normalized_upper.startswith("SELECT"):
            return False, "Only SELECT queries are allowed"

        if any(keyword in normalized_upper for keyword in ("DROP ", "DELETE ", "INSERT ", "UPDATE ", "ALTER ", "CREATE ", "ATTACH ", "DETACH ", "PRAGMA ")):
            return False, "Mutation and administrative statements are blocked"

        tables = set(self._extract_referenced_tables(normalized))
        disallowed = tables - self.NL_SQL_ALLOWED_TABLES
        if disallowed:
            return False, f"Query references tables outside the allowlist: {sorted(disallowed)}"

        if "SENSOR_STREAMS" in normalized_upper:
            if "WHERE" not in normalized_upper:
                return False, "Queries on sensor_streams require a WHERE clause"
            if "MACHINE_ID" not in normalized_upper and "TIMESTAMP" not in normalized_upper:
                return False, "Queries on sensor_streams must filter by machine_id or timestamp"

        return True, ""

    async def nl_to_sql(self, question: str, llm_client: Any = None) -> Dict[str, Any]:
        from config.settings import NL_TO_SQL_PROMPT

        if llm_client is None:
            from llm.llm_client import LLMClient
            llm_client = LLMClient()

        schema = await self.async_get_table_summary()
        prompt = NL_TO_SQL_PROMPT.format(schema=schema, question=question)
        system_prompt = (
            "You are a SQL expert for a Coiltech automation platform database. "
            "Generate only valid SQLite SELECT queries. No explanations."
        )

        try:
            generated_sql = await llm_client.generate(
                prompt=prompt, system_prompt=system_prompt, model_type="routing"
            )
            generated_sql = generated_sql.strip()
            if generated_sql.startswith("```"):
                lines = generated_sql.split("\n")
                generated_sql = "\n".join(l for l in lines if not l.startswith("```"))
            generated_sql = generated_sql.strip()

            is_valid, error_message = self._validate_safe_select(generated_sql)
            if not is_valid:
                return {
                    "success": False,
                    "error": error_message,
                    "generated_sql": generated_sql,
                    "results": None,
                }

            results = await self.async_execute_query(generated_sql)
            return {
                "success": True, "question": question,
                "generated_sql": generated_sql, "results": results,
                "row_count": len(results) if isinstance(results, list) else 0,
            }
        except Exception as e:
            logger.error("NL to SQL failed", question=question, error=str(e))
            return {"success": False, "question": question, "error": str(e),
                    "generated_sql": None, "results": None}

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------
    def initialize_schema(self, schema_path: Optional[str] = None) -> None:
        if schema_path is None:
            schema_path = Path(__file__).parent / "schema.sql"
        if not os.path.exists(schema_path):
            logger.warning("Schema file not found", path=schema_path)
            return
        conn = self.connect()
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        try:
            conn.executescript(schema_sql)
            conn.commit()
            logger.info("Database schema initialized", path=schema_path)
            self._schema_cache = None
        except sqlite3.Error as e:
            logger.error("Schema initialization failed", error=str(e))
            raise

    # ------------------------------------------------------------------
    # Domain-specific query helpers
    # ------------------------------------------------------------------
    async def get_product_families(self, category: Optional[str] = None) -> List[Dict]:
        q = "SELECT * FROM product_families WHERE is_active = 1"
        params = None
        if category:
            q += " AND category = ?"
            params = (category,)
        q += " ORDER BY name"
        return await self.async_execute_query(q, params)

    async def get_product_models(self, family_id: Optional[int] = None) -> List[Dict]:
        q = "SELECT pm.*, pf.name AS family_name FROM product_models pm JOIN product_families pf ON pm.family_id = pf.id WHERE pm.is_active = 1"
        params = None
        if family_id:
            q += " AND pm.family_id = ?"
            params = (family_id,)
        q += " ORDER BY pf.name, pm.name"
        return await self.async_execute_query(q, params)

    async def get_projects(self, status: Optional[str] = None, customer_id: Optional[int] = None) -> List[Dict]:
        q = "SELECT * FROM v_project_status WHERE 1=1"
        params = []
        if status:
            q += " AND status = ?"
            params.append(status)
        if customer_id:
            q += " AND customer_id = ?"
            params.append(customer_id)
        q += " ORDER BY target_completion"
        return await self.async_execute_query(q, tuple(params) if params else None)

    async def get_project_detail(self, project_id: int) -> Optional[Dict[str, Any]]:
        project = await self.async_execute_query(
            """
            SELECT p.*, c.name AS customer_name, q.code AS quotation_code
            FROM projects p
            JOIN customers c ON p.customer_id = c.id
            LEFT JOIN quotations q ON p.quotation_id = q.id
            WHERE p.id = ?
            """,
            (project_id,),
            fetch_one=True,
        )
        if not project:
            return None

        milestones = await self.async_execute_query(
            "SELECT * FROM project_milestones WHERE project_id = ? ORDER BY planned_date",
            (project_id,),
        )
        machines = await self.async_execute_query(
            """
            SELECT pmach.*, pm.model_code, pm.name AS model_name
            FROM project_machines pmach
            LEFT JOIN product_models pm ON pmach.model_id = pm.id
            WHERE pmach.project_id = ?
            ORDER BY pmach.id
            """,
            (project_id,),
        )
        configurations = await self.async_execute_query(
            "SELECT * FROM line_configurations WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        )
        project["milestones"] = milestones
        project["machines"] = machines
        project["line_configurations"] = configurations
        return project

    async def get_quotations(self, status: Optional[str] = None, customer_id: Optional[int] = None) -> List[Dict]:
        q = """SELECT q.*, c.name AS customer_name
               FROM quotations q JOIN customers c ON q.customer_id = c.id WHERE 1=1"""
        params = []
        if status:
            q += " AND q.status = ?"
            params.append(status)
        if customer_id:
            q += " AND q.customer_id = ?"
            params.append(customer_id)
        q += " ORDER BY q.created_at DESC"
        return await self.async_execute_query(q, tuple(params) if params else None)

    async def get_quotation_detail(self, quotation_id: int) -> Optional[Dict[str, Any]]:
        quotation = await self.async_execute_query(
            """
            SELECT q.*, c.name AS customer_name, po.code AS opportunity_code, po.title AS opportunity_title
            FROM quotations q
            JOIN customers c ON q.customer_id = c.id
            LEFT JOIN project_opportunities po ON q.opportunity_id = po.id
            WHERE q.id = ?
            """,
            (quotation_id,),
            fetch_one=True,
        )
        if not quotation:
            return None

        items = await self.async_execute_query(
            """
            SELECT qi.*, pm.model_code, pm.name AS model_name
            FROM quotation_items qi
            LEFT JOIN product_models pm ON qi.model_id = pm.id
            WHERE qi.quotation_id = ?
            ORDER BY qi.item_no
            """,
            (quotation_id,),
        )
        quotation["items"] = items
        return quotation

    async def get_installed_base(self, customer_id: Optional[int] = None) -> List[Dict]:
        q = "SELECT * FROM v_installed_base_overview WHERE 1=1"
        params = []
        if customer_id:
            q += " AND customer_id = ?"
            params.append(customer_id)
        q += " ORDER BY installation_date DESC"
        return await self.async_execute_query(q, tuple(params) if params else None)

    async def get_installed_machine_detail(self, installed_base_id: int) -> Optional[Dict[str, Any]]:
        machine = await self.async_execute_query(
            """
            SELECT ib.*, c.name AS customer_name, pm.model_code, pm.name AS model_name, pf.name AS family_name
            FROM installed_base ib
            JOIN customers c ON ib.customer_id = c.id
            LEFT JOIN product_models pm ON ib.model_id = pm.id
            LEFT JOIN product_families pf ON pm.family_id = pf.id
            WHERE ib.id = ?
            """,
            (installed_base_id,),
            fetch_one=True,
        )
        if not machine:
            return None

        machine["maintenance_logs"] = await self.get_maintenance_history(installed_base_id, limit=10)
        machine["open_tickets"] = await self.async_execute_query(
            "SELECT * FROM v_open_service_tickets WHERE installed_base_id = ? ORDER BY reported_date DESC",
            (installed_base_id,),
        )
        machine["active_alerts"] = await self.get_active_alerts(installed_base_id)
        return machine

    async def get_open_service_tickets(self, customer_id: Optional[int] = None) -> List[Dict]:
        q = "SELECT * FROM v_open_service_tickets WHERE 1=1"
        params = []
        if customer_id:
            q += " AND customer_id = ?"
            params.append(customer_id)
        q += " ORDER BY reported_date DESC"
        return await self.async_execute_query(q, tuple(params) if params else None)

    async def get_service_tickets(
        self,
        status: Optional[str] = None,
        customer_id: Optional[int] = None,
        installed_base_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        q = """
            SELECT st.*, c.name AS customer_name, ib.code AS machine_code, ib.machine_name
            FROM service_tickets st
            JOIN customers c ON st.customer_id = c.id
            LEFT JOIN installed_base ib ON st.installed_base_id = ib.id
            WHERE 1=1
        """
        params: List[Any] = []
        if status:
            q += " AND st.status = ?"
            params.append(status)
        if customer_id:
            q += " AND st.customer_id = ?"
            params.append(customer_id)
        if installed_base_id:
            q += " AND st.installed_base_id = ?"
            params.append(installed_base_id)
        q += " ORDER BY st.reported_date DESC, st.priority"
        return await self.async_execute_query(q, tuple(params) if params else None)

    async def get_spare_parts_low_stock(self) -> List[Dict]:
        return await self.async_execute_query("SELECT * FROM v_spare_parts_low_stock")

    async def get_spare_parts(
        self,
        category: Optional[str] = None,
        low_stock_only: bool = False,
    ) -> List[Dict[str, Any]]:
        if low_stock_only:
            q = "SELECT * FROM v_spare_parts_low_stock WHERE 1=1"
            params: List[Any] = []
            if category:
                q += " AND category = ?"
                params.append(category)
            return await self.async_execute_query(q, tuple(params) if params else None)

        q = """
            SELECT sp.*, sps.location, sps.quantity, sps.min_stock, sps.max_stock, sps.last_restock_date
            FROM spare_parts sp
            LEFT JOIN spare_part_stock sps ON sp.id = sps.part_id
            WHERE 1=1
        """
        params = []
        if category:
            q += " AND sp.category = ?"
            params.append(category)
        q += " ORDER BY sp.category, sp.name"
        return await self.async_execute_query(q, tuple(params) if params else None)

    async def get_active_alerts(self, machine_id: Optional[int] = None) -> List[Dict]:
        q = "SELECT * FROM v_active_alerts WHERE 1=1"
        params = []
        if machine_id:
            q += " AND machine_id = ?"
            params.append(machine_id)
        q += " ORDER BY timestamp DESC"
        return await self.async_execute_query(q, tuple(params) if params else None)

    async def get_recent_sensor_streams(
        self,
        machine_id: Optional[int] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        q = """
            SELECT ss.*, ib.code AS machine_code, ib.machine_name
            FROM sensor_streams ss
            JOIN installed_base ib ON ss.machine_id = ib.id
            WHERE ss.timestamp >= DATETIME('now', ?)
        """
        params: List[Any] = [f"-{hours} hours"]
        if machine_id:
            q += " AND ss.machine_id = ?"
            params.append(machine_id)
        q += " ORDER BY ss.timestamp DESC LIMIT ?"
        params.append(limit)
        return await self.async_execute_query(q, tuple(params))

    async def get_maintenance_history(self, installed_base_id: int, limit: int = 20) -> List[Dict]:
        q = """SELECT ml.*, ib.machine_name, ib.code AS machine_code
               FROM maintenance_logs ml
               JOIN installed_base ib ON ml.installed_base_id = ib.id
               WHERE ml.installed_base_id = ?
               ORDER BY ml.performed_date DESC LIMIT ?"""
        return await self.async_execute_query(q, (installed_base_id, limit))

    async def get_pipeline_summary(self) -> List[Dict]:
        return await self.async_execute_query("SELECT * FROM v_pipeline_summary")

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
        async with self.get_connection() as conn:
            cursor = await conn.execute(
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
            await conn.commit()
            return {"affected_rows": cursor.rowcount, "id": cursor.lastrowid}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    async def async_close(self) -> None:
        if self._async_connection:
            await self._async_connection.close()
            self._async_connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        await self.async_connect()
        return self

    async def __aexit__(self, *args):
        await self.async_close()
