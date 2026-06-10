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
    """
    Watches directories for new files and triggers RAG ingestion.
    
    Tracks processed files using a hash-based system to avoid
    duplicate processing.
    """
    
    def __init__(
        self,
        watch_dirs: list[str],
        rag_pipeline: Any,
        allowed_extensions: list[str],
        default_department: str = "general",
        scan_interval: int = 5,
        cag_cache: Any = None,
    ):
        """
        Initialize the file watcher.
        
        Args:
            watch_dirs: List of directories to watch.
            rag_pipeline: RAG pipeline instance for ingestion.
            allowed_extensions: List of allowed file extensions.
            default_department: Default department for files.
            scan_interval: Seconds between directory scans.
        """
        self.watch_dirs = [Path(d) for d in watch_dirs if d and os.path.exists(d)]
        self.rag_pipeline = rag_pipeline
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions]
        self.default_department = default_department
        self.scan_interval = scan_interval
        self._cag_cache = cag_cache

        # Track processed files by their hash
        self._processed_files: Dict[str, str] = {}  # path -> hash
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Department detection keywords
        self._department_keywords = {
            "production": ["üretim", "production", "manufacturing", "output", "hat", "line"],
            "maintenance": ["bakım", "maintenance", "repair", "onarım", "service", "arıza"],
            "quality": ["kalite", "quality", "inspection", "denetim", "test", "qa", "qc"],
            "logistics": ["lojistik", "logistics", "stok", "inventory", "depo", "warehouse"],
            "management": ["yönetim", "management", "rapor", "report", "kpi", "özet"],
        }
        
        logger.info(
            "FileWatcher initialized",
            watch_dirs=[str(d) for d in self.watch_dirs],
            extensions=self.allowed_extensions
        )
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for change detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _detect_department(self, file_path: Path) -> str:
        """
        Detect department from file name or path.
        
        Args:
            file_path: Path to the file.
        
        Returns:
            Detected department name.
        """
        # Check file name and parent directory names
        search_text = f"{file_path.stem} {file_path.parent.name}".lower()
        
        for dept, keywords in self._department_keywords.items():
            for keyword in keywords:
                if keyword in search_text:
                    logger.debug(f"Detected department {dept} from '{search_text}'")
                    return dept
        
        return self.default_department
    
    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        if not file_path.is_file():
            return False
        
        ext = file_path.suffix.lower()
        if ext not in self.allowed_extensions:
            return False
        
        # Skip hidden files and temp files
        if file_path.name.startswith('.') or file_path.name.startswith('~'):
            return False
        
        return True
    
    def _scan_directory(self, directory: Path) -> list[Path]:
        """Scan directory for valid files recursively."""
        files = []
        try:
            for item in directory.rglob("*"):
                if self._is_valid_file(item):
                    files.append(item)
        except Exception as e:
            logger.warning(f"Error scanning {directory}: {e}")
        return files
    
    async def _process_file(self, file_path: Path) -> bool:
        """
        Process a single file - ingest into RAG.
        
        Args:
            file_path: Path to the file.
        
        Returns:
            True if successfully processed.
        """
        try:
            # Calculate hash to check if already processed
            file_hash = self._get_file_hash(file_path)
            str_path = str(file_path)
            
            # Check if file was already processed with same hash
            if str_path in self._processed_files:
                if self._processed_files[str_path] == file_hash:
                    return False  # Already processed, no change
            
            # Detect department
            department = self._detect_department(file_path)
            
            logger.info(
                "Processing new file for RAG",
                file=file_path.name,
                department=department
            )
            
            # Ingest into RAG
            metadata = {
                "source": str(file_path),
                "original_filename": file_path.name,
                "auto_ingested": True,
                "ingested_at": datetime.utcnow().isoformat(),
                "watch_dir": str(file_path.parent)
            }
            
            result = await self.rag_pipeline.ingest_async(
                file_path=str(file_path),
                department=department,
                metadata=metadata
            )
            
            # Mark as processed
            self._processed_files[str_path] = file_hash

            # CAG: invalidate cached context for the affected collection
            if self._cag_cache is not None:
                self._cag_cache.invalidate_collection(department)

            logger.info(
                "File ingested successfully",
                file=file_path.name,
                department=department,
                chunks=result.get("chunks_stored", 0),
            )

            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    async def _watch_loop(self):
        """Main watch loop - scans directories periodically."""
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
            
            # Wait before next scan
            await asyncio.sleep(self.scan_interval)
        
        logger.info("File watcher stopped")
    
    async def start(self):
        """Start the file watcher."""
        if self._running:
            return
        
        if not self.watch_dirs:
            logger.warning("No valid watch directories configured")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("File watcher started", dirs=[str(d) for d in self.watch_dirs])
    
    async def stop(self):
        """Stop the file watcher."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("File watcher stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get watcher status."""
        return {
            "running": self._running,
            "watch_dirs": [str(d) for d in self.watch_dirs],
            "processed_files_count": len(self._processed_files),
            "scan_interval": self.scan_interval,
            "default_department": self.default_department
        }
    
    def get_processed_files(self) -> list[Dict[str, str]]:
        """Get list of processed files."""
        return [
            {"path": path, "hash": hash_val}
            for path, hash_val in self._processed_files.items()
        ]


class OpenWebUIWatcher(FileWatcher):
    """
    Specialized file watcher for Open WebUI integration.
    
    Automatically finds Open WebUI data directories and watches
    for uploaded files.
    """
    
    # Common Open WebUI data locations
    COMMON_PATHS = [
        # Windows
        os.path.expanduser("~/.open-webui/data/uploads"),
        os.path.expanduser("~/open-webui/data/uploads"),
        os.path.expanduser("~/.open_webui/uploads"),
        # Docker volumes (if mounted)
        "/app/backend/data/uploads",
        "./data/uploads",
        "../open-webui/data/uploads",
    ]
    
    def __init__(
        self,
        custom_path: Optional[str],
        rag_pipeline: Any,
        allowed_extensions: list[str],
        **kwargs
    ):
        """
        Initialize Open WebUI watcher.
        
        Args:
            custom_path: Custom path to Open WebUI uploads (optional).
            rag_pipeline: RAG pipeline instance.
            allowed_extensions: Allowed file extensions.
            **kwargs: Additional arguments for FileWatcher.
        """
        # Build list of directories to watch
        watch_dirs = []
        
        # Add custom path first if provided
        if custom_path and os.path.exists(custom_path):
            watch_dirs.append(custom_path)
            logger.info(f"Using custom Open WebUI path: {custom_path}")
        
        # Check common paths
        for path in self.COMMON_PATHS:
            if os.path.exists(path) and path not in watch_dirs:
                watch_dirs.append(path)
                logger.info(f"Found Open WebUI uploads at: {path}")
        
        super().__init__(
            watch_dirs=watch_dirs,
            rag_pipeline=rag_pipeline,
            allowed_extensions=allowed_extensions,
            **kwargs
        )
    
    @classmethod
    def find_openwebui_paths(cls) -> list[str]:
        """Find existing Open WebUI data paths."""
        found = []
        for path in cls.COMMON_PATHS:
            if os.path.exists(path):
                found.append(path)
        return found
