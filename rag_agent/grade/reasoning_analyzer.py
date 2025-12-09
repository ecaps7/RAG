"""LLM-based reasoning analyzer implementation."""

from __future__ import annotations

import json
from typing import List, Dict, Optional

from ..core.types import ContextChunk
from ..llm import llm_services
from ..utils.logging import get_logger
from ..utils.tracing import traceable_step
from .base import BaseReasoningAnalyzer, ReasoningResult
from pydantic import BaseModel, Field


class ReasoningAnalysisOutput(BaseModel):
    """Output schema for reasoning analysis."""
    information_sufficient: bool = Field(description="Whether the information is sufficient to answer the question")
    need_followup: bool = Field(description="Whether follow-up questions are needed")
    web_search_needed: bool = Field(description="Whether web search is needed")


class LLMReasoningAnalyzer(BaseReasoningAnalyzer):
    """LLM-based reasoning analyzer that assesses information sufficiency."""
    
    def __init__(self, trace_id: Optional[str] = None):
        """Initialize the LLM-based reasoning analyzer.
        
        Args:
            trace_id: Optional trace ID for logging
        """
        self.logger = get_logger(self.__class__.__name__, trace_id)
        # Use unified LLM service with structured output
        self._model = llm_services.get_structured_model(ReasoningAnalysisOutput)
    
    def _format_chunk_for_analysis(self, chunk: ContextChunk) -> Dict[str, str]:
        """Format a context chunk for analysis."""
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
    
    def _generate_analysis_prompt(self, question: str, chunks: List[ContextChunk]) -> str:
        """Generate the analysis prompt for the LLM."""
        formatted_chunks = [self._format_chunk_for_analysis(chunk) for chunk in chunks]
        
        prompt = """
        You are an expert reasoning analyzer for a Retrieval-Augmented Generation (RAG) system.
        Your task is to evaluate whether the retrieved context chunks contain sufficient information to answer the user's question.
        
        For the given user question and retrieved chunks, you must:
        1. Determine if the information is sufficient to answer the question completely
        2. Determine if follow-up questions are needed to get more specific information
        3. Determine if web search is needed to get additional information not available in the chunks
        
        Decision Guidelines:
        
        - Information is sufficient if:
          - The chunks contain all necessary information to answer the question completely
          - The information is relevant, accurate, and covers all aspects of the question
        
        - Follow-up is needed if:
          - The chunks contain some relevant information but are missing specific details
          - The question is ambiguous and needs clarification
          - More context is needed to provide a complete answer
        
        - Web search is needed if:
          - The chunks contain little or no relevant information
          - The question requires up-to-date information not available in the chunks
          - The question requires external knowledge not present in the chunks
        
        User Question:
        {question}
        
        Retrieved Chunks:
        {chunks}
        """
        
        return prompt.format(
            question=question,
            chunks=json.dumps(formatted_chunks, ensure_ascii=False, indent=2)
        )
    
    @traceable_step("reasoning_analysis", run_type="llm")
    def analyze(self, question: str, chunks: List[ContextChunk]) -> ReasoningResult:
        """Analyze the sufficiency of information to answer the question using LLM.
        
        Args:
            question: The user question
            chunks: List of retrieved context chunks
            
        Returns:
            Reasoning result indicating information sufficiency, need for follow-up, etc.
        """
        try:
            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(question, chunks)
            
            # Get LLM response with structured output
            messages = [
                {"role": "system", "content": "You are an expert reasoning analyzer."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._model.invoke(messages)
            
            # Map to ReasoningResult object
            return ReasoningResult(
                information_sufficient=response.information_sufficient,
                need_followup=response.need_followup,
                web_search_needed=response.web_search_needed
            )
            
        except Exception as e:
            self.logger.error(f"Reasoning analysis failed: {e}")
            
            # Return default result in case of error
            return ReasoningResult(
                information_sufficient=len(chunks) > 0,
                need_followup=len(chunks) == 0,
                web_search_needed=False
            )