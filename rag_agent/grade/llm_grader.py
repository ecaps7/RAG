"""LLM-based chunk grader implementation."""

from __future__ import annotations

import json
from typing import List, Dict, Optional

from ..core.types import ContextChunk
from ..llm import llm_services, GradeResult as GradeResultSchema
from ..utils.logging import get_logger
from ..utils.tracing import traceable_step
from .base import BaseGrader, GradeResult


class LLMBasedGrader(BaseGrader):
    """LLM-based chunk grader that uses a language model to assess relevance."""
    
    def __init__(self, relevance_threshold: float = 0.5, trace_id: Optional[str] = None):
        """Initialize the LLM-based grader.
        
        Args:
            relevance_threshold: Threshold for considering a chunk relevant (0.0-1.0)
            trace_id: Optional trace ID for logging
        """
        self.relevance_threshold = relevance_threshold
        self.logger = get_logger(self.__class__.__name__, trace_id)
        # Use unified LLM service with structured output
        self._model = llm_services.get_structured_model(GradeResultSchema)
    
    def _format_chunk_for_grading(self, chunk: ContextChunk) -> Dict[str, str]:
        """Format a context chunk for grading."""
        # Get source type from metadata if available
        source_type = chunk.metadata.get('retrieval_source', 'unknown')
        if chunk.source_id == 'sql_database':
            source_type = 'sql'
        
        # Truncate content if too long (keep first 1500 chars)
        content = chunk.content[:1500] + ('...' if len(chunk.content) > 1500 else '')
        
        return {
            "id": chunk.id,
            "content": content,
            "source_type": source_type,
            "title": chunk.title or "",
            "similarity": round(chunk.similarity, 3)
        }
    
    def _generate_grading_prompt(self, question: str, chunks: List[ContextChunk]) -> str:
        """Generate the grading prompt for the LLM."""
        formatted_chunks = [self._format_chunk_for_grading(chunk) for chunk in chunks]
        
        prompt = """
You are an expert relevance grader for a Retrieval-Augmented Generation (RAG) system.
Your task is to evaluate whether each retrieved context chunk is relevant to the user's question.

For each chunk, you must:
1. Determine its relevance to the question on a scale from 0.0 to 1.0
2. Provide clear reasoning for your assessment
3. Indicate if the chunk should be included in the context for answer generation

Relevance Guidelines:
- 1.0: Directly answers the question or provides essential information
- 0.7-0.9: Highly relevant, provides significant supporting information
- 0.4-0.6: Moderately relevant, provides some useful context
- 0.1-0.3: Slightly relevant, contains minimal useful information
- 0.0: Not relevant at all

User Question:
{question}

Retrieved Chunks:
{chunks}
""".format(
            question=question,
            chunks=json.dumps(formatted_chunks, ensure_ascii=False, indent=2)
        )
        
        return prompt
    
    @traceable_step("chunk_grading", run_type="llm")
    def grade(self, question: str, chunks: List[ContextChunk]) -> List[GradeResult]:
        """Grade relevance of retrieved chunks using LLM.
        
        Args:
            question: The user question
            chunks: List of retrieved context chunks
            
        Returns:
            List of grade results with relevance scores and reasoning
        """
        if not chunks:
            return []
        
        try:
            # Generate grading prompt
            prompt = self._generate_grading_prompt(question, chunks)
            
            # Get LLM response with structured output
            messages = [
                {"role": "system", "content": "You are an expert relevance grader."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._model.invoke(messages)
            
            # Map to GradeResult objects directly from structured output
            grade_results = []
            chunk_map = {chunk.id: chunk for chunk in chunks}
            
            for result in response.results:
                chunk = chunk_map.get(result.chunk_id)
                
                if chunk:
                    grade_results.append(GradeResult(
                        chunk_id=result.chunk_id,
                        relevance_score=result.relevance_score,
                        is_relevant=result.is_relevant,
                        reasoning=result.reasoning,
                        source_type=result.source_type
                    ))
            
            # Ensure we have results for all chunks
            if len(grade_results) < len(chunks):
                self.logger.warning(f"LLM returned grades for only {len(grade_results)} of {len(chunks)} chunks")
                
                # Add default grades for missing chunks
                graded_chunk_ids = {result.chunk_id for result in grade_results}
                for chunk in chunks:
                    if chunk.id not in graded_chunk_ids:
                        # Determine source type
                        source_type = chunk.metadata.get('retrieval_source', 'unknown')
                        if chunk.source_id == 'sql_database':
                            source_type = 'sql'
                        
                        # Default to low relevance
                        grade_results.append(GradeResult(
                            chunk_id=chunk.id,
                            relevance_score=0.3,
                            is_relevant=False,
                            reasoning="LLM did not provide a grade for this chunk. Defaulting to low relevance.",
                            source_type=source_type
                        ))
            
            return grade_results
            
        except Exception as e:
            self.logger.error(f"Grading failed: {e}")
            
            # Return default grades in case of error
            grade_results = []
            for chunk in chunks:
                source_type = chunk.metadata.get('retrieval_source', 'unknown')
                if chunk.source_id == 'sql_database':
                    source_type = 'sql'
                
                grade_results.append(GradeResult(
                    chunk_id=chunk.id,
                    relevance_score=0.5,  # Neutral default
                    is_relevant=True,     # Default to including the chunk
                    reasoning="Grading failed. Defaulting to neutral relevance.",
                    source_type=source_type
                ))
            
            return grade_results
    
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
            # Consider chunk relevant if either:
            # 1. LLM explicitly marked it as relevant, OR
            # 2. Relevance score is above the threshold
            if grade_result.is_relevant or grade_result.relevance_score >= self.relevance_threshold:
                # Update the chunk's similarity score with the relevance score
                chunk.similarity = grade_result.relevance_score
                # Add grading metadata
                chunk.metadata['relevance_score'] = str(round(grade_result.relevance_score, 3))
                chunk.metadata['grading_reasoning'] = grade_result.reasoning[:200]  # Truncate if too long
                relevant_chunks.append(chunk)
        
        # Sort by relevance score in descending order
        relevant_chunks.sort(key=lambda x: x.similarity, reverse=True)
        
        return relevant_chunks
