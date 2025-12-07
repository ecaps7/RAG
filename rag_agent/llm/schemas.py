"""Output schemas for LLM structured output."""

from pydantic import BaseModel, Field
from typing import List, Optional


class AnswerOutput(BaseModel):
    """Answer generation module output schema."""
    answer: str = Field(description="Answer content")
    citations: List[str] = Field(description="List of citation sources")
    confidence: float = Field(description="Answer confidence score (0.0-1.0)")


class GradeResultItem(BaseModel):
    """Single relevance grade result."""
    chunk_id: str = Field(description="Context chunk ID")
    relevance_score: float = Field(description="Relevance score (0.0-1.0)")
    is_relevant: bool = Field(description="Whether the chunk is relevant")
    reasoning: str = Field(description="Reasoning for the grade")
    source_type: str = Field(description="Source type of the chunk")


class GradeResult(BaseModel):
    """Batch relevance grade results."""
    results: List[GradeResultItem] = Field(description="List of grade results")
