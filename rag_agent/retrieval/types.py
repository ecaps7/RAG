from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class SearchResult:
    """A search result from a retrieval source."""
    id: str
    content: str
    score: float
    source: str # "vector", "bm25", "sql", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SQLResult:
    """A SQL query result."""
    metric_name: str
    metric_value: Any
    unit: str
    stock_code: str
    company_name: str
    report_period: str
    source_table_id: str