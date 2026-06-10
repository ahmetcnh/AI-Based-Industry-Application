"""
Excel Analyzer Agent.

Provides deterministic pandas-based analysis for uploaded .xlsx and .csv files.
This is separate from RAG: it reads the full table and performs real tabular
operations such as sum, average, filtering, grouping, and top-N selection.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SUPPORTED_OPERATIONS = {"summary", "sum", "mean", "count", "min", "max", "top", "filter"}


@dataclass
class ExcelAnalysisResult:
    file_name: str
    operation: str
    sheet_names: List[str]
    selected_sheet: Optional[str]
    question: Optional[str]
    result: Dict[str, Any]
    preview: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "operation": self.operation,
            "sheet_names": self.sheet_names,
            "selected_sheet": self.selected_sheet,
            "question": self.question,
            "result": self.result,
            "preview": self.preview,
            "warnings": self.warnings,
        }


class ExcelAnalyzerAgent:
    """Analyze Excel/CSV files with pandas instead of semantic retrieval."""

    def analyze(
        self,
        file_path: str,
        question: Optional[str] = None,
        sheet_name: Optional[str] = None,
        operation: Optional[str] = None,
        value_column: Optional[str] = None,
        group_by: Optional[str] = None,
        filter_column: Optional[str] = None,
        filter_value: Optional[str] = None,
        top_n: int = 10,
    ) -> ExcelAnalysisResult:
        path = Path(file_path)
        frames = self._load_frames(path)
        sheet_names = list(frames.keys())
        warnings: List[str] = []

        if sheet_name:
            resolved_sheet = self._resolve_sheet(sheet_name, sheet_names)
            if resolved_sheet is None:
                raise ValueError(f"Sheet not found: {sheet_name}. Available sheets: {sheet_names}")
            frames = {resolved_sheet: frames[resolved_sheet]}
            selected_sheet = resolved_sheet
        else:
            selected_sheet = None
            if len(frames) > 1:
                warnings.append("No sheet_name supplied; analysis was run on all sheets.")

        resolved_operation = self._infer_operation(question, operation)
        if resolved_operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported operation: {resolved_operation}. Supported: {sorted(SUPPORTED_OPERATIONS)}")

        per_sheet: Dict[str, Any] = {}
        preview: List[Dict[str, Any]] = []
        for current_sheet, df in frames.items():
            analysis = self._analyze_frame(
                df=df,
                sheet_name=current_sheet,
                question=question,
                operation=resolved_operation,
                value_column=value_column,
                group_by=group_by,
                filter_column=filter_column,
                filter_value=filter_value,
                top_n=top_n,
                warnings=warnings,
            )
            per_sheet[current_sheet] = analysis
            if not preview:
                preview = analysis.get("preview", [])

        return ExcelAnalysisResult(
            file_name=path.name,
            operation=resolved_operation,
            sheet_names=sheet_names,
            selected_sheet=selected_sheet,
            question=question,
            result={"sheets": per_sheet},
            preview=preview,
            warnings=warnings,
        )

    def _load_frames(self, path: Path) -> Dict[str, pd.DataFrame]:
        ext = path.suffix.lower()
        if ext == ".xlsx":
            return pd.read_excel(path, sheet_name=None)
        if ext == ".csv":
            return {"CSV": pd.read_csv(path)}
        raise ValueError("Only .xlsx and .csv files are supported by Excel analysis.")

    def _analyze_frame(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        question: Optional[str],
        operation: str,
        value_column: Optional[str],
        group_by: Optional[str],
        filter_column: Optional[str],
        filter_value: Optional[str],
        top_n: int,
        warnings: List[str],
    ) -> Dict[str, Any]:
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]

        resolved_value = self._resolve_column(df, value_column) if value_column else self._infer_value_column(df, question)
        resolved_group = self._resolve_column(df, group_by) if group_by else self._infer_group_column(df, question, resolved_value)
        resolved_filter_col = self._resolve_column(df, filter_column) if filter_column else None
        filtered = self._apply_filter(df, resolved_filter_col, filter_value) if resolved_filter_col else df

        numeric_columns = list(filtered.select_dtypes(include="number").columns)
        base: Dict[str, Any] = {
            "sheet_name": sheet_name,
            "row_count": int(len(df)),
            "filtered_row_count": int(len(filtered)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
            "numeric_columns": numeric_columns,
            "value_column": resolved_value,
            "group_by": resolved_group,
            "filter": {"column": resolved_filter_col, "value": filter_value} if resolved_filter_col else None,
            "preview": self._records(filtered.head(top_n)),
        }

        if operation == "summary":
            base["summary"] = self._summary(filtered)
            return base

        if operation == "filter":
            if not resolved_filter_col:
                warnings.append("Filter operation requested but filter_column was not supplied.")
            base["rows"] = self._records(filtered.head(top_n))
            return base

        if operation == "count":
            if resolved_group:
                grouped = filtered.groupby(resolved_group, dropna=False).size().reset_index(name="count")
                grouped = grouped.sort_values("count", ascending=False).head(top_n)
                base["grouped"] = self._records(grouped)
            elif resolved_value:
                base["count"] = int(filtered[resolved_value].count())
            else:
                base["count"] = int(len(filtered))
            return base

        target_columns = self._target_numeric_columns(filtered, resolved_value)
        if not target_columns:
            warnings.append("No numeric columns found for numeric operation.")
            base["aggregation"] = {}
            return base

        if operation == "top":
            sort_col = resolved_value or target_columns[0]
            top_rows = filtered.sort_values(sort_col, ascending=False).head(top_n)
            base["top_rows"] = self._records(top_rows)
            return base

        if resolved_group:
            grouped = getattr(filtered.groupby(resolved_group, dropna=False)[target_columns], operation)().reset_index()
            sort_col = target_columns[0]
            ascending = operation == "min"
            grouped = grouped.sort_values(sort_col, ascending=ascending).head(top_n)
            base["grouped"] = self._records(grouped)
        else:
            base["aggregation"] = {
                col: self._json_safe(getattr(filtered[col], operation)())
                for col in target_columns
            }
        return base

    def _summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        missing = df.isna().sum().to_dict()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        numeric = df.select_dtypes(include="number")
        numeric_summary = numeric.describe().to_dict() if not numeric.empty else {}
        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "dtypes": self._json_safe(dtypes),
            "missing_values": self._json_safe(missing),
            "numeric_summary": self._json_safe(numeric_summary),
        }

    def _target_numeric_columns(self, df: pd.DataFrame, value_column: Optional[str]) -> List[str]:
        if value_column:
            if not pd.api.types.is_numeric_dtype(df[value_column]):
                raise ValueError(f"Column is not numeric: {value_column}")
            return [value_column]
        return list(df.select_dtypes(include="number").columns)

    def _apply_filter(
        self,
        df: pd.DataFrame,
        filter_column: Optional[str],
        filter_value: Optional[str],
    ) -> pd.DataFrame:
        if not filter_column or filter_value is None:
            return df

        series = df[filter_column]
        numeric_value = self._parse_number(filter_value)
        if numeric_value is not None and pd.api.types.is_numeric_dtype(series):
            return df[series == numeric_value]

        needle = self._normalize(str(filter_value))
        mask = series.astype(str).map(lambda item: needle in self._normalize(item))
        return df[mask]

    def _infer_operation(self, question: Optional[str], operation: Optional[str]) -> str:
        if operation:
            normalized = operation.strip().lower()
            aliases = {"avg": "mean", "average": "mean", "total": "sum", "describe": "summary"}
            return aliases.get(normalized, normalized)

        normalized_question = self._normalize(question or "")
        if any(word in normalized_question for word in ["top", "ilk", "en yuksek", "largest", "highest"]):
            return "top"
        if any(word in normalized_question for word in ["ortalama", "average", "avg", "mean"]):
            return "mean"
        if any(word in normalized_question for word in ["toplam", "total", "sum"]):
            return "sum"
        if any(word in normalized_question for word in ["minimum", "min", "en dusuk", "lowest"]):
            return "min"
        if any(word in normalized_question for word in ["maximum", "max", "en yuksek", "highest"]):
            return "max"
        if any(word in normalized_question for word in ["kac", "count", "adet", "sayisi", "number of"]):
            return "count"
        if any(word in normalized_question for word in ["filter", "filtre", "sadece", "where"]):
            return "filter"
        return "summary"

    def _infer_value_column(self, df: pd.DataFrame, question: Optional[str]) -> Optional[str]:
        normalized_question = self._normalize(question or "")
        numeric_columns = list(df.select_dtypes(include="number").columns)
        for col in numeric_columns:
            if self._normalize(col) in normalized_question:
                return col
        if len(numeric_columns) == 1:
            return numeric_columns[0]
        return None

    def _infer_group_column(
        self,
        df: pd.DataFrame,
        question: Optional[str],
        value_column: Optional[str],
    ) -> Optional[str]:
        normalized_question = self._normalize(question or "")
        has_group_signal = any(word in normalized_question for word in [" by ", " per ", "gore", "bazinda", "group"])
        if not has_group_signal:
            return None
        for col in df.columns:
            if col == value_column:
                continue
            if self._normalize(col) in normalized_question:
                return col
        categorical = [col for col in df.columns if col != value_column and not pd.api.types.is_numeric_dtype(df[col])]
        return categorical[0] if categorical else None

    def _resolve_sheet(self, requested: str, sheet_names: List[str]) -> Optional[str]:
        normalized = self._normalize(requested)
        for sheet in sheet_names:
            if self._normalize(sheet) == normalized:
                return sheet
        return None

    def _resolve_column(self, df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
        if not requested:
            return None
        normalized = self._normalize(requested)
        for col in df.columns:
            if self._normalize(col) == normalized:
                return col
        for col in df.columns:
            if normalized in self._normalize(col) or self._normalize(col) in normalized:
                return col
        raise ValueError(f"Column not found: {requested}. Available columns: {list(df.columns)}")

    def _records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        return self._json_safe(df.to_dict(orient="records"))

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (datetime, date, pd.Timestamp)):
            return value.isoformat()
        if hasattr(value, "item"):
            try:
                return self._json_safe(value.item())
            except Exception:
                pass
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return value

    def _parse_number(self, raw: str) -> Optional[float]:
        try:
            return float(str(raw).replace(",", "."))
        except ValueError:
            return None

    def _normalize(self, value: Any) -> str:
        raw = str(value or "").casefold()
        decomposed = unicodedata.normalize("NFKD", raw)
        ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
        return f" {ascii_text.strip()} "
