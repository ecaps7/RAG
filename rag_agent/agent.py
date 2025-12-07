"""RAG Agent - main pipeline orchestrator."""

from __future__ import annotations

import time
from typing import Optional, Iterator, List, Tuple

from .core.types import Answer, CitationInfo
from .generation import AnswerGenerator
from .grade import LLMBasedGrader
from .retrieval import LocalRetriever
from .utils.logging import get_logger
from .utils.tracing import trace_pipeline, trace_pipeline_stream


class RagAgent:
    """Main RAG Agent that orchestrates the retrieval and generation pipeline.
    
    Flow:
    1. Hybrid Retrieval (Vector + BM25 + SQL with RRF fusion)
    2. Relevance Grading
    3. Generate answer with citations
    """
    
    def __init__(
        self,
        retriever: Optional[LocalRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        grader: Optional[LLMBasedGrader] = None,
        trace_id: Optional[str] = None,
    ):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        self.retriever = retriever or LocalRetriever(trace_id=trace_id)
        self.generator = generator or AnswerGenerator()
        self.grader = grader or LLMBasedGrader(trace_id=trace_id)

    @trace_pipeline("rag_pipeline")
    def run(self, question: str) -> Answer:
        """Run the full RAG pipeline and return an answer.
        
        Args:
            question: The user's question
            
        Returns:
            Answer object with text, citations, confidence, and metadata
        """
        self.logger.debug(f"Processing question: {question}")
        
        # Step 1: Hybrid Retrieval (Vector + BM25 + SQL with RRF fusion)
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, 8)
        t1 = time.perf_counter()
        self.logger.debug(f"Retrieved {len(chunks)} chunks in {t1 - t0:.2f} seconds")
        
        # Step 2: Relevance Grading (RAG 3.0 improvement)
        t0 = time.perf_counter()
        relevant_chunks = self.grader.filter_relevant(question, chunks)
        t1 = time.perf_counter()
        self.logger.info(f"Grading completed: {len(relevant_chunks)} relevant chunks out of {len(chunks)} total in {t1 - t0:.2f} seconds")
        
        # Step 3: Generation
        t0 = time.perf_counter()
        answer = self.generator.generate(question, relevant_chunks)
        t1 = time.perf_counter()
        
        answer.meta.update({
            "chunks_retrieved": str(len(chunks)),
            "chunks_filtered": str(len(relevant_chunks)),
        })
        
        self.logger.debug(f"Generated answer with confidence {answer.confidence:.2f} in {t1 - t0:.2f} seconds")
        return answer

    @trace_pipeline_stream("rag_pipeline_stream")
    def run_stream(self, question: str) -> Tuple[Iterator[str], List[CitationInfo]]:
        """Run the pipeline and return a streaming answer iterator.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (text_stream_iterator, citation_info_list)
            The citation_info_list contains numbered references that can be filtered
            based on which [n] markers appear in the generated text.
        """
        self.logger.debug(f"Processing streaming question: {question}")
        
        # Step 1: Hybrid Retrieval
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, 8)
        t1 = time.perf_counter()
        self.logger.debug(f"Retrieved {len(chunks)} chunks in {t1 - t0:.2f} seconds")
        
        # Step 2: Relevance Grading (RAG 3.0 improvement)
        t0 = time.perf_counter()
        relevant_chunks = self.grader.filter_relevant(question, chunks)
        t1 = time.perf_counter()
        self.logger.info(f"Grading completed: {len(relevant_chunks)} relevant chunks out of {len(chunks)} total in {t1 - t0:.2f} seconds")
        
        # Step 3: Stream generation
        stream = self.generator.stream_answer_text(question, relevant_chunks)

        # Build citation info with numbered references
        citation_infos: List[CitationInfo] = []
        for idx, ch in enumerate(relevant_chunks, start=1):
            title = ch.citation or ch.title or ch.source_id or ""
            # 从 metadata 中提取 page 和 doctype
            metadata = ch.metadata or {}
            page = str(metadata.get("page", ""))
            doc_type = metadata.get("doctype", "text")
            # SQL 结果特殊处理
            if ch.source_id == "sql_database":
                doc_type = "sql"
            
            citation_infos.append(CitationInfo(
                ref=idx,
                title=str(title),
                source_id=str(ch.source_id),
                source_type=ch.source_type,
                doc_type=doc_type,
                page=page,
                score=ch.similarity,
                reliability=ch.reliability,
            ))

        return stream, citation_infos
