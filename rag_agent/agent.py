"""RAG Agent - main pipeline orchestrator."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Iterator, List, Tuple, Callable

from .config import TOP_K
from .core.types import Answer, Intent, ContextChunk
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

        local_chunks: List[ContextChunk] = []
        web_chunks: List[ContextChunk] = []

        # Step 3: Parallel Retrieval
        local_chunks, web_chunks = self._parallel_retrieve(
            question, plan, use_local=plan.use_local, use_web=plan.use_web
        )

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

        local_chunks: List[ContextChunk] = []
        web_chunks: List[ContextChunk] = []

        # Step 3: Parallel Retrieval
        local_chunks, web_chunks = self._parallel_retrieve(
            question, plan, use_local=plan.use_local, use_web=plan.use_web
        )

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

    def _parallel_retrieve(
        self,
        question: str,
        plan,
        use_local: bool = True,
        use_web: bool = True,
    ) -> Tuple[List[ContextChunk], List[ContextChunk]]:
        """Execute local and web retrieval in parallel.
        
        Args:
            question: The user's question
            plan: The retrieval plan with top_k settings
            use_local: Whether to use local retrieval
            use_web: Whether to use web retrieval
            
        Returns:
            Tuple of (local_chunks, web_chunks)
        """
        local_chunks: List[ContextChunk] = []
        web_chunks: List[ContextChunk] = []
        
        # If only one source is needed, run directly without threading overhead
        if use_local and not use_web:
            t0 = time.perf_counter()
            local_chunks = self.local.retrieve(question, plan.local_top_k)
            t1 = time.perf_counter()
            self._debug.print_local_retrieval(local_chunks, question, duration_ms=(t1 - t0) * 1000)
            return local_chunks, web_chunks
        
        if use_web and not use_local:
            t0 = time.perf_counter()
            web_chunks = self.web.retrieve(question, plan.web_top_k)
            t1 = time.perf_counter()
            self._debug.print_web_retrieval(web_chunks, question, duration_ms=(t1 - t0) * 1000)
            return local_chunks, web_chunks
        
        if not use_local and not use_web:
            return local_chunks, web_chunks
        
        # Both sources needed - run in parallel
        # Get current LangSmith run tree to pass as parent to child threads
        parent_run = None
        try:
            from langsmith.run_helpers import get_current_run_tree
            parent_run = get_current_run_tree()
        except Exception:
            pass
        
        local_result: List[ContextChunk] = []
        web_result: List[ContextChunk] = []
        local_duration_ms = 0.0
        web_duration_ms = 0.0
        
        def do_local() -> Tuple[List[ContextChunk], float]:
            # Create a child trace under the parent run
            from .utils.tracing import _is_langsmith_enabled
            try:
                if _is_langsmith_enabled() and parent_run is not None:
                    from langsmith import trace as langsmith_trace
                    with langsmith_trace(
                        name="local_retrieval",
                        run_type="retriever",
                        parent=parent_run,
                    ):
                        t0 = time.perf_counter()
                        chunks = self.local.retrieve(question, plan.local_top_k)
                        duration = (time.perf_counter() - t0) * 1000
                        return chunks, duration
            except Exception:
                pass
            # Fallback without tracing
            t0 = time.perf_counter()
            chunks = self.local.retrieve(question, plan.local_top_k)
            duration = (time.perf_counter() - t0) * 1000
            return chunks, duration
        
        def do_web() -> Tuple[List[ContextChunk], float]:
            # Create a child trace under the parent run
            from .utils.tracing import _is_langsmith_enabled
            try:
                if _is_langsmith_enabled() and parent_run is not None:
                    from langsmith import trace as langsmith_trace
                    with langsmith_trace(
                        name="web_retrieval",
                        run_type="retriever",
                        parent=parent_run,
                    ):
                        t0 = time.perf_counter()
                        chunks = self.web.retrieve(question, plan.web_top_k)
                        duration = (time.perf_counter() - t0) * 1000
                        return chunks, duration
            except Exception:
                pass
            # Fallback without tracing
            t0 = time.perf_counter()
            chunks = self.web.retrieve(question, plan.web_top_k)
            duration = (time.perf_counter() - t0) * 1000
            return chunks, duration
        
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(do_local): "local",
                executor.submit(do_web): "web",
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    result, duration = future.result()
                    if source == "local":
                        local_result = result
                        local_duration_ms = duration
                    else:
                        web_result = result
                        web_duration_ms = duration
                except Exception as e:
                    self.logger.warning("Parallel retrieval failed for %s: %s", source, e)
        
        t_total = (time.perf_counter() - t_start) * 1000
        
        # Debug output (sequential timing info for each source)
        self._debug.print_local_retrieval(local_result, question, duration_ms=local_duration_ms)
        self._debug.print_web_retrieval(web_result, question, duration_ms=web_duration_ms)
        self.logger.info(
            "Parallel retrieval completed: local=%d web=%d total=%.1fms (saved ~%.1fms)",
            len(local_result), len(web_result), t_total,
            max(0, local_duration_ms + web_duration_ms - t_total)
        )
        
        return local_result, web_result
