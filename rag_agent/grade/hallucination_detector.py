"""LLM-based hallucination detector implementation."""

from __future__ import annotations

import json
from typing import List, Dict, Optional

from ..core.types import ContextChunk
from ..llm import llm_services
from ..utils.logging import get_logger
from ..utils.tracing import traceable_step
from .base import BaseHallucinationDetector, HallucinationResult
from pydantic import BaseModel, Field


class HallucinationDetectionOutput(BaseModel):
    """Output schema for hallucination detection."""
    hallucination_detected: bool = Field(description="Whether hallucination is detected")
    confidence: float = Field(description="Confidence score of the detection (0.0-1.0)")
    reasoning: Optional[str] = Field(description="Reasoning for the detection, only provided if hallucination is detected")
    hallucinated_content: Optional[str] = Field(description="The hallucinated content if detected")


class LLMHallucinationDetector(BaseHallucinationDetector):
    """LLM-based hallucination detector that checks answer consistency with context."""
    
    def __init__(self, trace_id: Optional[str] = None):
        """Initialize the LLM-based hallucination detector.
        
        Args:
            trace_id: Optional trace ID for logging
        """
        self.logger = get_logger(self.__class__.__name__, trace_id)
        # Use unified LLM service with structured output
        self._model = llm_services.get_structured_model(HallucinationDetectionOutput)
    
    def _format_chunk_for_detection(self, chunk: ContextChunk) -> Dict[str, str]:
        """Format a context chunk for hallucination detection."""
        # Get source type from metadata if available
        source_type = chunk.metadata.get('retrieval_source', 'unknown')
        if chunk.source_id == 'sql_database':
            source_type = 'sql'
        
        # Truncate content if too long (keep first 1000 chars)
        content = chunk.content[:1000] + ('...' if len(chunk.content) > 1000 else '')
        
        return {
            "id": chunk.id,
            "content": content,
            "source_type": source_type,
            "title": chunk.title or "",
            "relevance_score": round(chunk.similarity, 3)
        }
    
    def _generate_detection_prompt(self, question: str, answer: str, chunks: List[ContextChunk]) -> str:
        """Generate the hallucination detection prompt for the LLM."""
        formatted_chunks = [self._format_chunk_for_detection(chunk) for chunk in chunks]
        
        prompt = """
        You are an expert hallucination detector for a Retrieval-Augmented Generation (RAG) system.
        Your task is to check if the generated answer contains any information that is not supported by the provided context chunks.
        
        For the given user question, generated answer, and context chunks, you must:
        1. Compare the answer with the context chunks line by line
        2. Identify any information in the answer that is not present in the context
        3. Determine if this constitutes a hallucination
        4. If hallucination is detected, provide a short reason indicating what information is missing or unsupported
        5. If hallucination is detected, extract the specific hallucinated content
        6. Assign a confidence score to your detection (0.0-1.0)
        7. If no hallucination is detected, set reasoning to None
        
        Hallucination Criteria:
        - Information that is completely absent from all context chunks
        - Information that contradicts the context chunks
        - Specific facts, figures, names, or dates not mentioned in the context
        - Unsupported claims or assertions
        
        Non-Hallucination Criteria:
        - Logical inferences based on the context
        - Paraphrasing of context information
        - General knowledge that is widely accepted and not in conflict with the context
        
        User Question:
        {question}
        
        Generated Answer:
        {answer}
        
        Context Chunks:
        {chunks}
        """
        
        return prompt.format(
            question=question,
            answer=answer,
            chunks=json.dumps(formatted_chunks, ensure_ascii=False, indent=2)
        )
    
    @traceable_step("hallucination_detection", run_type="llm")
    def detect(self, question: str, answer: str, chunks: List[ContextChunk]) -> HallucinationResult:
        """Detect hallucinations in the generated answer using LLM.
        
        Args:
            question: The user question
            answer: The generated answer
            chunks: List of context chunks used for generation
            
        Returns:
            Hallucination detection result
        """
        try:
            # Generate detection prompt
            prompt = self._generate_detection_prompt(question, answer, chunks)
            
            # Get LLM response with structured output
            messages = [
                {"role": "system", "content": "You are an expert hallucination detector."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._model.invoke(messages)
            
            # Map to HallucinationResult object
            return HallucinationResult(
                hallucination_detected=response.hallucination_detected,
                confidence=response.confidence,
                reasoning=response.reasoning,
                hallucinated_content=response.hallucinated_content
            )
            
        except Exception as e:
            self.logger.error(f"Hallucination detection failed: {e}")
            
            # Return default result in case of error
            return HallucinationResult(
                hallucination_detected=False,
                confidence=0.5,
                reasoning=None,
                hallucinated_content=None
            )