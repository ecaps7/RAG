from __future__ import annotations

from typing import Optional, Iterator, List, Tuple

from .common.config import TOP_K
from .common.logging import get_logger
from .common.types import Answer, Intent
from .llm.answer_generator import AnswerGenerator
from .retriever.local_retriever import LocalRetriever
from .retriever.web_retriever import WebRetriever
from .router.fusion_layer import FusionLayer
from .router.intent_classifier import IntentClassifier
from .router.retrieval_router import RetrievalRouter


class RagAgent:
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
        """执行检索与融合，并返回答案文本的流式迭代器与引用列表。"""
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