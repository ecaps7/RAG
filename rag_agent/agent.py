"""RAG Agent - main pipeline orchestrator."""

from __future__ import annotations

import time
from typing import Optional, Iterator, List, Tuple

from .config import TOP_K
from .core.types import Answer, Intent
from .generation import AnswerGenerator
from .intent import IntentClassifier, RetrievalRouter
from .retrieval import LocalRetriever, WebRetriever, FusionLayer
from .utils.logging import get_logger
from .utils.debug import get_debug_printer
from .utils.tracing import trace_pipeline, trace_pipeline_stream


class RagAgent:
    """Main RAG Agent that orchestrates the retrieval and generation pipeline.
    
    Flow:
    1. Classify intent of the question
    2. Route to appropriate retrievers based on intent
    3. Fuse results from multiple sources
    4. Generate answer with citations
    """
    
    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        router: Optional[RetrievalRouter] = None,
        local: Optional[LocalRetriever] = None,
        web: Optional[WebRetriever] = None,
        fusion: Optional[FusionLayer] = None,
        generator: Optional[AnswerGenerator] = None,
        trace_id: Optional[str] = None,
    ):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        self.intent_classifier = intent_classifier or IntentClassifier(trace_id)
        self.router = router or RetrievalRouter()
        self.local = local or LocalRetriever(trace_id=trace_id)
        self.web = web or WebRetriever(trace_id=trace_id)
        self.fusion = fusion or FusionLayer(trace_id)
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
        
        # Step 1: Intent Classification
        t0 = time.perf_counter()
        intent, intent_conf = self.intent_classifier.classify(question)
        t1 = time.perf_counter()
        self._debug.print_intent(intent, intent_conf, duration_ms=(t1 - t0) * 1000)
        
        # Step 2: Retrieval Routing
        plan = self.router.plan(intent)
        self._debug.print_routing(plan, intent)

        local_chunks = []
        web_chunks = []

        # Step 3: Retrieval
        if plan.use_local:
            t0 = time.perf_counter()
            local_chunks = self.local.retrieve(question, plan.local_top_k)
            t1 = time.perf_counter()
            self._debug.print_local_retrieval(local_chunks, question, duration_ms=(t1 - t0) * 1000)
        if plan.use_web:
            t0 = time.perf_counter()
            web_chunks = self.web.retrieve(question, plan.web_top_k)
            t1 = time.perf_counter()
            self._debug.print_web_retrieval(web_chunks, question, duration_ms=(t1 - t0) * 1000)

        # Step 4: Fusion
        t0 = time.perf_counter()
        fusion = self.fusion.aggregate(local_chunks, web_chunks, intent, k=TOP_K["fusion"])
        t1 = time.perf_counter()
        self._debug.print_fusion(fusion, len(local_chunks), len(web_chunks), duration_ms=(t1 - t0) * 1000)
        
        # Step 5: Generation
        t0 = time.perf_counter()
        answer = self.generator.generate(question, fusion)
        t1 = time.perf_counter()
        
        answer.meta.update({
            "intent": intent.value,
            "intent_confidence": f"{intent_conf:.2f}",
            "local_chunks": str(len(local_chunks)),
            "web_chunks": str(len(web_chunks)),
        })
        
        # Debug: Print generation result and summary
        self._debug.print_generation(answer, len(fusion.selected_chunks), duration_ms=(t1 - t0) * 1000)
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
        plan = self.router.plan(intent)
        self._debug.print_routing(plan, intent)

        local_chunks = []
        web_chunks = []

        # Step 3: Retrieval
        if plan.use_local:
            t0 = time.perf_counter()
            local_chunks = self.local.retrieve(question, plan.local_top_k)
            t1 = time.perf_counter()
            self._debug.print_local_retrieval(local_chunks, question, duration_ms=(t1 - t0) * 1000)
        if plan.use_web:
            t0 = time.perf_counter()
            web_chunks = self.web.retrieve(question, plan.web_top_k)
            t1 = time.perf_counter()
            self._debug.print_web_retrieval(web_chunks, question, duration_ms=(t1 - t0) * 1000)

        # Step 4: Fusion
        t0 = time.perf_counter()
        fusion = self.fusion.aggregate(local_chunks, web_chunks, intent, k=TOP_K["fusion"])
        t1 = time.perf_counter()
        self._debug.print_fusion(fusion, len(local_chunks), len(web_chunks), duration_ms=(t1 - t0) * 1000)
        
        # Step 5: Stream generation
        stream = self.generator.stream_answer_text(question, fusion)

        citations: List[str] = []
        for ch in fusion.selected_chunks:
            cite = ch.citation or ch.title or ch.source_id
            if cite:
                citations.append(str(cite))

        return stream, citations
