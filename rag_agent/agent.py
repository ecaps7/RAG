"""RAG Agent - main pipeline orchestrator."""

from __future__ import annotations

import time
from typing import Optional, Iterator, List, Tuple

from .core.types import Answer, CitationInfo
from .core.state_machine import app
from .generation import AnswerGenerator
from .retrieval import LocalRetriever
from .utils.logging import get_logger
from .utils.tracing import trace_pipeline, trace_pipeline_stream


class RagAgent:
    """Main RAG Agent that orchestrates the retrieval and generation pipeline.
    
    Flow:
    1. 调用LangGraph状态机处理
    2. 支持并行检索、迭代检索和多跳
    3. 生成答案并进行幻觉检测
    """
    
    def __init__(
        self,
        trace_id: Optional[str] = None,
    ):
        self.logger = get_logger(self.__class__.__name__, trace_id)
        self.app = app
        self.generator = AnswerGenerator()
        self.retriever = LocalRetriever(trace_id=trace_id)

    @trace_pipeline("rag_pipeline")
    def run(self, question: str) -> Answer:
        """Run the full RAG pipeline and return an answer.
        
        Args:
            question: The user's question
            
        Returns:
            Answer object with text, citations, confidence, and metadata
        """
        self.logger.debug(f"Processing question: {question}")
        
        # 调用LangGraph状态机
        from .core.types import TerminationInfo, RetrievalCache
        
        initial_state = {
            # 问题与迭代控制
            "original_question": question,
            "current_query": question,
            "current_hop": 0,
            "max_hops": 5,
            
            # 增量记忆
            "accumulated_context": [],
            "context_id_set": set(),
            "entity_history": set(),
            "query_history": [],
            
            # 缺口分析
            "missing_info": [],
            "information_sufficient": False,
            "need_followup": False,
            "web_search_needed": False,
            
            # 终止控制
            "info_gain_log": [],
            "consecutive_no_gain": 0,
            "termination": TerminationInfo(
                should_terminate=False,
                reason="",
                details=""
            ),
            
            # 检索缓存
            "retrieval_cache": RetrievalCache(),
            
            # 检索完成标志
            "fusion_retrieval_done": False,
            "sql_retrieval_done": False,
            
            # 生成与评估
            "generation": None,
            "hallucination_detected": False,
            
            # 最终输出
            "final_answer": None,
        }
        
        t0 = time.perf_counter()
        result = self.app.invoke(initial_state)
        t1 = time.perf_counter()
        
        self.logger.debug(f"Generated answer with confidence {result['final_answer'].confidence:.2f} in {t1 - t0:.2f} seconds")
        return result["final_answer"]

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
        
        # 调用LangGraph状态机获取结果
        from .core.types import TerminationInfo, RetrievalCache
        
        initial_state = {
            # 问题与迭代控制
            "original_question": question,
            "current_query": question,
            "current_hop": 0,
            "max_hops": 1,
            
            # 增量记忆
            "accumulated_context": [],
            "context_id_set": set(),
            "entity_history": set(),
            "query_history": [],
            
            # 缺口分析
            "missing_info": [],
            "information_sufficient": False,
            "need_followup": False,
            "web_search_needed": False,
            
            # 终止控制
            "info_gain_log": [],
            "consecutive_no_gain": 0,
            "termination": TerminationInfo(
                should_terminate=False,
                reason="",
                details=""
            ),
            
            # 检索缓存
            "retrieval_cache": RetrievalCache(),
            
            # 检索完成标志
            "fusion_retrieval_done": False,
            "sql_retrieval_done": False,
            
            # 生成与评估
            "generation": None,
            "hallucination_detected": False,
            
            # 最终输出
            "final_answer": None,
        }
        
        t0 = time.perf_counter()
        result = self.app.invoke(initial_state)
        t1 = time.perf_counter()
        
        self.logger.debug(f"Retrieved and processed context in {t1 - t0:.2f} seconds")
        
        # 构建流式输出
        stream = self.generator.stream_answer_text(question, result["accumulated_context"])
        
        # 构建引用信息
        citation_infos: List[CitationInfo] = []
        for idx, ch in enumerate(result["accumulated_context"], start=1):
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
