"""RAG Agent - main pipeline orchestrator."""

from __future__ import annotations

from typing import Optional, Iterator, List, Tuple

from .config import TOP_K
from .core.types import Answer, Intent
from .generation import AnswerGenerator
from .intent import IntentClassifier, RetrievalRouter
from .retrieval import LocalRetriever, WebRetriever, FusionLayer
from .utils.logging import get_logger


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

    def run(self, question: str) -> Answer:
        """Run the full RAG pipeline and return an answer.
        
        Args:
            question: The user's question
            
        Returns:
            Answer object with text, citations, confidence, and metadata
        """
        intent, intent_conf = self.intent_classifier.classify(question)
        plan = self.router.plan(intent)

        local_chunks = []
        web_chunks = []

        if plan.use_local:
            local_chunks = self.local.retrieve(question, plan.local_top_k)
        if plan.use_web:
            web_chunks = self.web.retrieve(question, plan.web_top_k)

        fusion = self.fusion.aggregate(local_chunks, web_chunks, intent, k=TOP_K["fusion"])
        answer = self.generator.generate(question, fusion)
        
        answer.meta.update({
            "intent": intent.value,
            "intent_confidence": f"{intent_conf:.2f}",
            "local_chunks": str(len(local_chunks)),
            "web_chunks": str(len(web_chunks)),
        })
        
        self.logger.info("Answer confidence=%.2f", answer.confidence)
        return answer

    def run_stream(self, question: str) -> Tuple[Iterator[str], List[str]]:
        """Run the pipeline and return a streaming answer iterator.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (text_stream_iterator, citations_list)
        """
        intent, _intent_conf = self.intent_classifier.classify(question)
        plan = self.router.plan(intent)

        local_chunks = []
        web_chunks = []

        if plan.use_local:
            local_chunks = self.local.retrieve(question, plan.local_top_k)
        if plan.use_web:
            web_chunks = self.web.retrieve(question, plan.web_top_k)

        fusion = self.fusion.aggregate(local_chunks, web_chunks, intent, k=TOP_K["fusion"])
        stream = self.generator.stream_answer_text(question, fusion)

        citations: List[str] = []
        for ch in fusion.selected_chunks:
            cite = ch.citation or ch.title or ch.source_id
            if cite:
                citations.append(str(cite))

        return stream, citations
