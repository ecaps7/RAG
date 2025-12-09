"""LangGraph 节点实现"""

from typing import List, Optional
from rag_agent.core.state import AgentState
from rag_agent.core.types import ContextChunk, Answer
from rag_agent.retrieval.retrieval_manager import RetrievalManager
from rag_agent.retrieval.engine import LocalRetriever
from rag_agent.generation.generator import AnswerGenerator
from rag_agent.grade.llm_grader import LLMBasedGrader
from rag_agent.retrieval.types import SearchResult
from rag_agent.utils.logging import get_logger

# 初始化日志记录器
logger = get_logger(__name__)

# 初始化组件
retrieval_manager = RetrievalManager()
answer_generator = AnswerGenerator()
llm_grader = LLMBasedGrader()

# 导入并初始化查询重写和追问生成组件
from rag_agent.query import QueryRewriter, FollowupQueryGenerator
query_rewriter = QueryRewriter()
followup_generator = FollowupQueryGenerator()

# 导入并初始化推理分析器和幻觉检测器
from rag_agent.grade import LLMReasoningAnalyzer, LLMHallucinationDetector
reasoning_analyzer = LLMReasoningAnalyzer()
hallucination_detector = LLMHallucinationDetector()

# ================= 节点实现 =================

def query_rewrite_node(state: AgentState) -> dict:
    """查询重写/分解节点"""
    logger.info(f"[query_rewrite_node] 开始处理查询: {state['question']}")
    result = query_rewriter.rewrite_query(state)
    logger.info(f"[query_rewrite_node] 查询重写完成，新查询: {result.get('question', state['question'])}")
    return result

def sql_generation_execution_node(state: AgentState) -> dict:
    """SQL生成与执行节点 - 可并行"""
    # logger.info(f"[sql_generation_execution_node] 开始处理查询: {state['question']}")
    if retrieval_manager.should_route_to_sql(state["question"]):
        logger.info(f"[sql_generation_execution_node] 查询需要路由到SQL执行")
        sql_results = retrieval_manager.execute_sql_query(state["question"])
        logger.info(f"[sql_generation_execution_node] SQL执行完成，结果: {sql_results[:100]}..." if sql_results else "[sql_generation_execution_node] SQL执行完成，无结果")
    else:
        # logger.info(f"[sql_generation_execution_node] 查询不需要路由到SQL执行")
        sql_results = None
    return {"sql_results": sql_results}

def vector_retrieval_node(state: AgentState) -> dict:
    """向量检索节点 - 可并行"""
    logger.info(f"[vector_retrieval_node] 开始向量检索，查询: {state['question']}")
    vector_results = retrieval_manager.vector_retrieve(state["question"], top_k=10)
    logger.info(f"[vector_retrieval_node] 向量检索完成，找到 {len(vector_results)} 个结果")
    # 转换为ContextChunk格式
    chunks = retrieval_manager.local_retriever._to_chunks(vector_results)
    logger.info(f"[vector_retrieval_node] 结果转换完成，共 {len(chunks)} 个上下文块")
    return {"vector_results": chunks}

def bm25_retrieval_node(state: AgentState) -> dict:
    """BM25检索节点 - 可并行"""
    logger.info(f"[bm25_retrieval_node] 开始BM25检索，查询: {state['question']}")
    bm25_results = retrieval_manager.bm25_retrieve(state["question"], top_k=10)
    logger.info(f"[bm25_retrieval_node] BM25检索完成，找到 {len(bm25_results)} 个结果")
    # 转换为ContextChunk格式
    chunks = retrieval_manager.local_retriever._to_chunks(bm25_results)
    logger.info(f"[bm25_retrieval_node] 结果转换完成，共 {len(chunks)} 个上下文块")
    return {"bm25_results": chunks}

def rrf_fusion_node(state: AgentState) -> dict:
    """RRF融合节点"""
    logger.info(f"[rrf_fusion_node] 开始RRF融合，向量结果数: {len(state['vector_results'])}, BM25结果数: {len(state['bm25_results'])}")
    result_lists = []
    
    # 准备向量检索结果
    if state["vector_results"]:
        logger.info(f"[rrf_fusion_node] 准备向量检索结果，共 {len(state['vector_results'])} 个")
        vector_search_results = [
            SearchResult(
                id=chunk.id,
                content=chunk.content,
                score=chunk.similarity,
                source="vector",
                metadata=chunk.metadata
            ) for chunk in state["vector_results"]
        ]
        result_lists.append(vector_search_results)
    
    # 准备BM25检索结果
    if state["bm25_results"]:
        logger.info(f"[rrf_fusion_node] 准备BM25检索结果，共 {len(state['bm25_results'])} 个")
        bm25_search_results = [
            SearchResult(
                id=chunk.id,
                content=chunk.content,
                score=chunk.similarity,
                source="bm25",
                metadata=chunk.metadata
            ) for chunk in state["bm25_results"]
        ]
        result_lists.append(bm25_search_results)
    
    # 执行RRF融合
    if result_lists:
        logger.info(f"[rrf_fusion_node] 开始执行RRF融合，融合 {len(result_lists)} 个结果列表")
        fused_search_results = retrieval_manager.rrf_fusion(result_lists)
        logger.info(f"[rrf_fusion_node] RRF融合完成，获得 {len(fused_search_results)} 个融合结果")
        chunks = retrieval_manager.local_retriever._to_chunks(fused_search_results)
        fused_results = chunks
    else:
        logger.info(f"[rrf_fusion_node] 没有可用结果进行融合")
        fused_results = []
    logger.info(f"[rrf_fusion_node] RRF融合节点完成，最终结果数: {len(fused_results)}")
    return {"fused_results": fused_results}

def rerank_node(state: AgentState) -> dict:
    """重排节点"""
    logger.info(f"[rerank_node] 开始重排，融合结果数: {len(state['fused_results'])}")
    if state["fused_results"]:
        # 转换为SearchResult格式
        logger.info(f"[rerank_node] 转换为SearchResult格式")
        search_results = [
            SearchResult(
                id=chunk.id,
                content=chunk.content,
                score=chunk.similarity,
                source=chunk.metadata.get("retrieval_source", "unknown"),
                metadata=chunk.metadata
            ) for chunk in state["fused_results"]
        ]
        # 执行重排
        logger.info(f"[rerank_node] 执行重排，top_k=8")
        reranked_search_results = retrieval_manager.rerank(state["question"], search_results, top_k=8)
        logger.info(f"[rerank_node] 重排完成，结果数: {len(reranked_search_results)}")
        chunks = retrieval_manager.local_retriever._to_chunks(reranked_search_results)
        reranked_results = chunks
    else:
        logger.info(f"[rerank_node] 没有融合结果需要重排")
        reranked_results = []
    logger.info(f"[rerank_node] 重排节点完成，最终结果数: {len(reranked_results)}")
    return {"reranked_results": reranked_results}

def context_aggregation_node(state: AgentState) -> dict:
    """上下文聚合节点 - 处理并行检索结果"""
    logger.info(f"[context_aggregation_node] 开始上下文聚合，SQL结果: {state['sql_results'] is not None}, 重排结果数: {len(state['reranked_results'])}")
    final_documents = []
    
    # 添加SQL结果（如果有）
    if state["sql_results"]:
        logger.info(f"[context_aggregation_node] 添加SQL结果")
        sql_chunk = ContextChunk(
            id="sql_structured_result",
            content=state["sql_results"],
            source_type="local",
            source_id="sql_database",
            title="结构化数据查询结果",
            similarity=1.0,
            recency=1.0,
            reliability=0.95,
            metadata={"doctype": "sql", "retrieval_source": "sql"},
        )
        final_documents.append(sql_chunk)
    
    # 添加重排结果
    logger.info(f"[context_aggregation_node] 添加重排结果，共 {len(state['reranked_results'])} 个")
    final_documents.extend(state["reranked_results"])
    
    # 累积上下文
    accumulated_context = state.get("accumulated_context", [])
    current_len = len(accumulated_context)
    accumulated_context.extend(final_documents)
    logger.info(f"[context_aggregation_node] 累积上下文完成，当前长度: {current_len} -> {len(accumulated_context)}")
    
    return {
        "final_documents": final_documents,
        "documents": final_documents,  # 兼容旧字段
        "accumulated_context": accumulated_context
    }

def reasoning_analyzer_node(state: AgentState) -> dict:
    """推理分析器节点"""
    logger.info(f"[reasoning_analyzer_node] 开始推理分析，文档数: {len(state['final_documents'])}, 重试次数: {state['retry_count']}/{state['max_retries']}")
    
    # 使用LLM推理分析器进行信息充足性判断
    result = reasoning_analyzer.analyze(
        question=state["question"],
        chunks=state["final_documents"]
    )
    
    logger.info(f"[reasoning_analyzer_node] 推理分析完成")
    logger.info(f"[reasoning_analyzer_node] 下一步操作: 信息充足={result.information_sufficient}, 需要追问={result.need_followup}, 需要联网搜索={result.web_search_needed}")
    
    return {
        "information_sufficient": result.information_sufficient,
        "need_followup": result.need_followup,
        "web_search_needed": result.web_search_needed
    }

def followup_query_generation_node(state: AgentState) -> dict:
    """追问生成节点"""
    logger.info(f"[followup_query_generation_node] 开始生成追问，当前问题: {state['question']}")
    result = followup_generator.generate_followup(state)
    logger.info(f"[followup_query_generation_node] 追问生成完成，追问数量: {len(result.get('followup_queries', []))}")
    return result

def answer_generation_node(state: AgentState) -> dict:
    """答案生成节点"""
    logger.info(f"[answer_generation_node] 开始生成答案，原始问题: {state['original_question']}")
    answer = answer_generator.generate(state["original_question"], state["accumulated_context"])
    logger.info(f"[answer_generation_node] 答案生成完成")
    return {"generation": answer.text}

def hallucination_detection_node(state: AgentState) -> dict:
    """幻觉检测节点"""
    logger.info(f"[hallucination_detection_node] 开始幻觉检测")
    
    # 使用LLM幻觉检测器进行幻觉检测
    result = hallucination_detector.detect(
        question=state["original_question"],
        answer=state["generation"],
        chunks=state["accumulated_context"]
    )
    
    logger.info(f"[hallucination_detection_node] 幻觉检测完成，检测结果: {'发现幻觉' if result.hallucination_detected else '未发现幻觉'}")
    logger.info(f"[hallucination_detection_node] 检测置信度: {result.confidence}, 推理: {result.reasoning[:100]}...")
    
    return {
        "hallucination_detected": result.hallucination_detected,
        "hallucination_confidence": result.confidence,
        "hallucination_reasoning": result.reasoning,
        "hallucinated_content": result.hallucinated_content
    }

def web_search_node(state: AgentState) -> dict:
    """联网搜索节点"""
    logger.info(f"[web_search_node] 开始联网搜索，问题: {state['question']}")
    # TODO: 实现联网搜索逻辑
    # 当前简化实现，直接返回
    logger.info(f"[web_search_node] 联网搜索完成")
    return {}

def final_output_node(state: AgentState) -> dict:
    """最终输出节点"""
    logger.info(f"[final_output_node] 开始生成最终答案，原始问题: {state['original_question']}")
    # 生成最终答案
    final_answer = answer_generator.generate(state["original_question"], state["accumulated_context"])
    logger.info(f"[final_output_node] 最终答案生成完成")
    return {"final_answer": final_answer}

# ================= 条件路由实现 =================

def reasoning_conditional(state: AgentState) -> str:
    """推理分析器条件路由"""
    if state["information_sufficient"]:
        return "answer_generation"
    elif state["web_search_needed"]:
        return "web_search"
    elif state["need_followup"]:
        return "followup_query_generation"
    else:
        return "web_search"

def hallucination_conditional(state: AgentState) -> str:
    """幻觉检测条件路由"""
    if state["hallucination_detected"]:
        return "followup_query_generation"
    else:
        return "final_output"

def aggregation_conditional(state: AgentState) -> str:
    """上下文聚合条件路由 - 检查是否同时存在SQL结果和重排结果"""
    sql_results_exist = state.get("sql_results") is not None
    reranked_results_exist = len(state.get("reranked_results", [])) > 0
    
    # 如果同时存在SQL结果和重排结果，则继续到推理分析器
    if sql_results_exist and reranked_results_exist:
        return "reasoning_analyzer"
    else:
        return "end"
