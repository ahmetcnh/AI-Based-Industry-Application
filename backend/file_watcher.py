"""
File Watcher Module for Open WebUI Integration.

Watches specified directories for new files and automatically ingests them
into the RAG system.
"""

import os
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, Set, Optional, Any, Callable
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class FileWatcher:
    def __init__(
        self,
        watch_dirs: list[str],
        rag_pipeline: Any,
        allowed_extensions: list[str],
        default_department: str = "general",
        scan_interval: int = 5,
        cag_cache: Any = None,
    ):
        self.watch_dirs = [Path(d) for d in watch_dirs if d and os.path.exists(d)]
        self.rag_pipeline = rag_pipeline
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions]
        self.default_department = default_department
        self.scan_interval = scan_interval
        self._cag_cache = cag_cache
        self._processed_files: Dict[str, str] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._department_keywords = {
            "production": ["üretim", "production", "manufacturing", "output", "hat", "line"],
            "maintenance": ["bakım", "maintenance", "repair", "onarım", "service", "arıza"],
            "quality": ["kalite", "quality", "inspection", "denetim", "test", "qa", "qc"],
            "logistics": ["lojistik", "logistics", "stok", "inventory", "depo", "warehouse"],
            "management": ["yönetim", "management", "rapor", "report", "kpi", "özet"],
        }
        logger.info("FileWatcher initialized", watch_dirs=[str(d) for d in self.watch_dirs], extensions=self.allowed_extensions)

    def _get_file_hash(self, file_path: Path) -> str:
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    def _detect_department(self, file_path: Path) -> str:
        search_text = f"{file_path.stem} {file_path.parent.name}".lower()
        for dept, keywords in self._department_keywords.items():
            for keyword in keywords:
                if keyword in search_text:
                    return dept
        return self.default_department

    def _is_valid_file(self, file_path: Path) -> bool:
        if not file_path.is_file():
            return False
        ext = file_path.suffix.lower()
        if ext not in self.allowed_extensions:
            return False
        if file_path.name.startswith('.') or file_path.name.startswith('~'):
            return False
        return True

    def _scan_directory(self, directory: Path) -> list[Path]:
        files = []
        try:
            for item in directory.rglob("*"):
                if self._is_valid_file(item):
                    files.append(item)
        except Exception as e:
            logger.warning(f"Error scanning {directory}: {e}")
        return files

    async def _process_file(self, file_path: Path) -> bool:
        try:
            file_hash = self._get_file_hash(file_path)
            str_path = str(file_path)
            if str_path in self._processed_files:
                if self._processed_files[str_path] == file_hash:
                    return False
            department = self._detect_department(file_path)
            logger.info("Processing new file for RAG", file=file_path.name, department=department)
            metadata = {
                "source": str(file_path),
                "original_filename": file_path.name,
                "auto_ingested": True,
                "ingested_at": datetime.utcnow().isoformat(),
                "watch_dir": str(file_path.parent),
            }
            # ingest_async internally calls maybe_create_agent(collection, tenant_id, user_id)
            # so the router agent is auto-created for any new collection after a successful ingest.
            result = await self.rag_pipeline.ingest_async(
                file_path=str(file_path),
                department=department,
                metadata=metadata,
                tenant_id=1,
                user_id=1,
            )
            self._processed_files[str_path] = file_hash
            if self._cag_cache is not None:
                self._cag_cache.invalidate_collection(department)
            logger.info("File ingested successfully", file=file_path.name, department=department, chunks=result.get("chunks_stored", 0))
            return True
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False

    async def scan_now(self) -> int:
        """Immediate one-off scan so a file attached to the current chat
        message is ingested without waiting for the next interval."""
        processed = 0
        for watch_dir in self.watch_dirs:
            if not watch_dir.exists():
                continue
            for file_path in self._scan_directory(watch_dir):
                if await self._process_file(file_path):
                    processed += 1
        return processed

    async def _watch_loop(self):
        logger.info("File watcher started")
        while self._running:
            try:
                for watch_dir in self.watch_dirs:
                    if not watch_dir.exists():
                        continue
                    files = self._scan_directory(watch_dir)
                    for file_path in files:
                        if not self._running:
                            break
                        await self._process_file(file_path)
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
            await asyncio.sleep(self.scan_interval)
        logger.info("File watcher stopped")

    async def start(self):
        if self._running:
            return
        if not self.watch_dirs:
            logger.warning("No valid watch directories configured")
            return
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("File watcher started", dirs=[str(d) for d in self.watch_dirs])

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("File watcher stopped")

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "watch_dirs": [str(d) for d in self.watch_dirs],
            "processed_files_count": len(self._processed_files),
            "scan_interval": self.scan_interval,
            "default_department": self.default_department,
        }
