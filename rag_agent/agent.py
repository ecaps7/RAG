"""RAG Agent - main pipeline orchestrator."""

from __future__ import annotations

import time
from typing import Optional, Iterator, List, Tuple

from .core.types import Answer, Intent, ContextChunk
from .generation import AnswerGenerator
from .intent import IntentClassifier, RetrievalRouter
from .retrieval import LocalRetriever
from .utils.logging import get_logger
from .utils.debug import get_debug_printer
from .utils.tracing import trace_pipeline, trace_pipeline_stream


class RagAgent:
    """Main RAG Agent that orchestrates the retrieval and generation pipeline.
    
    Flow:
    1. Classify intent of the question
    2. Route to appropriate retriever based on intent
    3. Retrieve and rank results (hybrid search)
    4. Generate answer with citations
    """
    
    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        router: Optional[RetrievalRouter] = None,
        retriever: Optional[LocalRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        trace_id: Optional[str] = None,
    ):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        self.intent_classifier = intent_classifier or IntentClassifier(trace_id)
        self.router = router or RetrievalRouter()
        self.retriever = retriever or LocalRetriever(trace_id=trace_id)
        self.generator = generator or AnswerGenerator()
        self._debug = get_debug_printer()

    @trace_pipeline("rag_pipeline")
    def run(self, question: str) -> Answer:
        """Run the full RAG pipeline and return an answer.
        
        Args:
            question: The user's question
            
        Returns:
            Answer object with text, citations, confidence, and metadata
        """
        # Debug: Print input question
        self._debug.print_question(question)
        
        # Step 1: Intent Classification (for future use)
        t0 = time.perf_counter()
        intent, intent_conf = self.intent_classifier.classify(question)
        t1 = time.perf_counter()
        self._debug.print_intent(intent, intent_conf, duration_ms=(t1 - t0) * 1000)
        
        # # Step 2: Retrieval Routing
        # plan = self.router.plan(intent, question)
        # self._debug.print_routing(plan, intent)

        # Step 3: Hybrid Retrieval (Vector + BM25 + SQL with RRF fusion)
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, 8)
        t1 = time.perf_counter()
        self._debug.print_local_retrieval(chunks, question, duration_ms=(t1 - t0) * 1000)
        
        # Step 4: Generation
        t0 = time.perf_counter()
        answer = self.generator.generate(question, chunks)
        t1 = time.perf_counter()
        
        answer.meta.update({
            "intent": intent.value,
            "intent_confidence": f"{intent_conf:.2f}",
            "chunks_retrieved": str(len(chunks)),
        })
        
        # Debug: Print generation result and summary
        self._debug.print_generation(answer, len(chunks), duration_ms=(t1 - t0) * 1000)
        self._debug.print_summary(answer, answer.meta)
        
        return answer

    @trace_pipeline_stream("rag_pipeline_stream")
    def run_stream(self, question: str) -> Tuple[Iterator[str], List[str]]:
        """Run the pipeline and return a streaming answer iterator.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (text_stream_iterator, citations_list)
        """
        # Debug: Print input question
        self._debug.print_question(question)
        
        # Step 1: Intent Classification
        t0 = time.perf_counter()
        intent, intent_conf = self.intent_classifier.classify(question)
        t1 = time.perf_counter()
        self._debug.print_intent(intent, intent_conf, duration_ms=(t1 - t0) * 1000)
        
        # Step 2: Retrieval Routing
        plan = self.router.plan(intent, question)
        self._debug.print_routing(plan, intent)

        # Step 3: Hybrid Retrieval
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, plan.local_top_k, intent=intent)
        t1 = time.perf_counter()
        self._debug.print_local_retrieval(chunks, question, duration_ms=(t1 - t0) * 1000)
        
        # Step 4: Stream generation
        stream = self.generator.stream_answer_text(question, chunks)

        # Extract citations from chunks
        citations: List[str] = []
        for ch in chunks:
            cite = ch.citation or ch.title or ch.source_id
            if cite:
                citations.append(str(cite))

        return stream, citations
