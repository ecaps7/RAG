"""Base classes and interfaces for chunk graders, reasoning analyzers, and hallucination detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ..core.types import ContextChunk


@dataclass
class GradeResult:
    """Result of grading a context chunk."""
    chunk_id: str
    relevance_score: float  # 0.0-1.0, higher means more relevant
    is_relevant: bool  # Whether the chunk is relevant to the question
    reasoning: str  # Why the chunk is relevant or not
    source_type: str  # The source type of the chunk (sql, vector, bm25)


@dataclass
class ReasoningResult:
    """Result of reasoning analysis."""
    information_sufficient: bool  # Whether the information is sufficient to answer the question
    need_followup: bool  # Whether follow-up questions are needed
    web_search_needed: bool  # Whether web search is needed
    missing_info: List = field(default_factory=list)  # List of missing information (MissingInfo objects)


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    hallucination_detected: bool  # Whether hallucination is detected
    confidence: float  # Confidence score of the detection (0.0-1.0)
    reasoning: Optional[str] = None  # Reasoning for the detection, only provided if hallucination is detected
    hallucinated_content: Optional[str] = None  # The hallucinated content if detected


class BaseGrader(ABC):
    """Base interface for all chunk graders."""
    
    @abstractmethod
    def grade(self, question: str, chunks: List[ContextChunk]) -> List[GradeResult]:
        """Grade relevance of retrieved chunks to the given question.
        
        Args:
            question: The user question
            chunks: List of retrieved context chunks
            
        Returns:
            List of grade results with relevance scores and reasoning
        """
        pass
    
    @abstractmethod
    def filter_relevant(self, question: str, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Filter chunks to only include those relevant to the question.
        
        Args:
            question: The user question
            chunks: List of retrieved context chunks
            
        Returns:
            Filtered list of relevant chunks
        """
        pass


class ChunkGrader(BaseGrader):
    """Default chunk grader implementation."""
    
    def __init__(self, relevance_threshold: float = 0.5):
        """Initialize the chunk grader.
        
        Args:
            relevance_threshold: Threshold for considering a chunk relevant (0.0-1.0)
        """
        self.relevance_threshold = relevance_threshold
    
    def filter_relevant(self, question: str, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Filter chunks to only include those relevant to the question.
        
        Args:
            question: The user question
            chunks: List of retrieved context chunks
            
        Returns:
            Filtered list of relevant chunks
        """
        grade_results = self.grade(question, chunks)
        relevant_chunks = []
        
        for chunk, grade_result in zip(chunks, grade_results):
            if grade_result.is_relevant:
                # Update the chunk's similarity score with the relevance score
                # This ensures the most relevant chunks are prioritized
                chunk.similarity = grade_result.relevance_score
                relevant_chunks.append(chunk)
        
        # Sort by relevance score in descending order
        relevant_chunks.sort(key=lambda x: x.similarity, reverse=True)
        
        return relevant_chunks


class BaseReasoningAnalyzer(ABC):
    """Base interface for reasoning analyzers."""
    
    @abstractmethod
    def analyze(self, question: str, chunks: List[ContextChunk]) -> ReasoningResult:
        """Analyze the sufficiency of information to answer the question.
        
        Args:
            question: The user question
            chunks: List of retrieved context chunks
            
        Returns:
            Reasoning result indicating information sufficiency, need for follow-up, etc.
        """
        pass


class BaseHallucinationDetector(ABC):
    """Base interface for hallucination detectors."""
    
    @abstractmethod
    def detect(self, question: str, answer: str, chunks: List[ContextChunk]) -> HallucinationResult:
        """Detect hallucinations in the generated answer.
        
        Args:
            question: The user question
            answer: The generated answer
            chunks: List of context chunks used for generation
            
        Returns:
            Hallucination detection result
        """
        pass
