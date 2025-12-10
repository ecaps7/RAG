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
    logger.info(f"[query_rewrite_node] 开始处理查询: {state['current_query']}")
    result = query_rewriter.rewrite_query(state)
    logger.info(f"[query_rewrite_node] 查询重写完成，新查询: {result.get('current_query', state['current_query'])}")
    return result

def sql_generation_execution_node(state: AgentState) -> dict:
    """SQL生成与执行节点 - 可并行"""
    if retrieval_manager.should_route_to_sql(state["current_query"]):
        logger.info(f"[sql_generation_execution_node] 查询需要路由到SQL执行")
        sql_results = retrieval_manager.execute_sql_query(state["current_query"])
        logger.info(f"[sql_generation_execution_node] SQL执行完成，结果: {sql_results[:100]}..." if sql_results else "[sql_generation_execution_node] SQL执行完成，无结果")
    else:
        logger.info(f"[sql_generation_execution_node] 查询不需要SQL，跳过")
        sql_results = None
    
    # 更新 retrieval_cache
    cache = state["retrieval_cache"]
    cache.sql_results = sql_results
    
    # 设置 SQL 检索完成标志
    return {
        "retrieval_cache": cache,
        "sql_retrieval_done": True
    }

def vector_retrieval_node(state: AgentState) -> dict:
    """向量检索节点 - 可并行"""
    logger.info(f"[vector_retrieval_node] 开始向量检索，查询: {state['current_query']}")
    vector_results = retrieval_manager.vector_retrieve(state["current_query"], top_k=10)
    logger.info(f"[vector_retrieval_node] 向量检索完成，找到 {len(vector_results)} 个结果")
    # 转换为ContextChunk格式
    chunks = retrieval_manager.local_retriever._to_chunks(vector_results)
    logger.info(f"[vector_retrieval_node] 结果转换完成，共 {len(chunks)} 个上下文块")
    
    # 更新 retrieval_cache
    cache = state["retrieval_cache"]
    cache.vector_results = chunks
    return {"retrieval_cache": cache}

def bm25_retrieval_node(state: AgentState) -> dict:
    """BM25检索节点 - 可并行"""
    logger.info(f"[bm25_retrieval_node] 开始BM25检索，查询: {state['current_query']}")
    bm25_results = retrieval_manager.bm25_retrieve(state["current_query"], top_k=10)
    logger.info(f"[bm25_retrieval_node] BM25检索完成，找到 {len(bm25_results)} 个结果")
    # 转换为ContextChunk格式
    chunks = retrieval_manager.local_retriever._to_chunks(bm25_results)
    logger.info(f"[bm25_retrieval_node] 结果转换完成，共 {len(chunks)} 个上下文块")
    
    # 更新 retrieval_cache
    cache = state["retrieval_cache"]
    cache.bm25_results = chunks
    return {"retrieval_cache": cache}

def rrf_fusion_node(state: AgentState) -> dict:
    """RRF融合节点"""
    cache = state["retrieval_cache"]
    logger.info(f"[rrf_fusion_node] 开始RRF融合，向量结果数: {len(cache.vector_results)}, BM25结果数: {len(cache.bm25_results)}")
    result_lists = []
    
    # 准备向量检索结果
    if cache.vector_results:
        logger.info(f"[rrf_fusion_node] 准备向量检索结果，共 {len(cache.vector_results)} 个")
        vector_search_results = [
            SearchResult(
                id=chunk.id,
                content=chunk.content,
                score=chunk.similarity,
                source="vector",
                metadata=chunk.metadata
            ) for chunk in cache.vector_results
        ]
        result_lists.append(vector_search_results)
    
    # 准备BM25检索结果
    if cache.bm25_results:
        logger.info(f"[rrf_fusion_node] 准备BM25检索结果，共 {len(cache.bm25_results)} 个")
        bm25_search_results = [
            SearchResult(
                id=chunk.id,
                content=chunk.content,
                score=chunk.similarity,
                source="bm25",
                metadata=chunk.metadata
            ) for chunk in cache.bm25_results
        ]
        result_lists.append(bm25_search_results)
    
    # 执行RRF融合
    if result_lists:
        logger.info(f"[rrf_fusion_node] 开始执行RRF融合，融合 {len(result_lists)} 个结果列表")
        fused_search_results = retrieval_manager.rrf_fusion(result_lists)
        logger.info(f"[rrf_fusion_node] RRF融合完成，获得 {len(fused_search_results)} 个融合结果")
        chunks = retrieval_manager.local_retriever._to_chunks(fused_search_results)
        cache.fused_results = chunks
    else:
        logger.info(f"[rrf_fusion_node] 没有可用结果进行融合")
        cache.fused_results = []
    
    logger.info(f"[rrf_fusion_node] RRF融合节点完成，最终结果数: {len(cache.fused_results)}")
    return {"retrieval_cache": cache}

def rerank_node(state: AgentState) -> dict:
    """重排节点"""
    cache = state["retrieval_cache"]
    logger.info(f"[rerank_node] 开始重排，融合结果数: {len(cache.fused_results)}")
    if cache.fused_results:
        # 转换为SearchResult格式
        logger.info(f"[rerank_node] 转换为SearchResult格式")
        search_results = [
            SearchResult(
                id=chunk.id,
                content=chunk.content,
                score=chunk.similarity,
                source=chunk.metadata.get("retrieval_source", "unknown"),
                metadata=chunk.metadata
            ) for chunk in cache.fused_results
        ]
        # 执行重排
        logger.info(f"[rerank_node] 执行重排，top_k=8")
        reranked_search_results = retrieval_manager.rerank(state["current_query"], search_results, top_k=8)
        logger.info(f"[rerank_node] 重排完成，结果数: {len(reranked_search_results)}")
        chunks = retrieval_manager.local_retriever._to_chunks(reranked_search_results)
        cache.reranked_results = chunks
    else:
        logger.info(f"[rerank_node] 没有融合结果需要重排")
        cache.reranked_results = []
    logger.info(f"[rerank_node] 重排节点完成，最终结果数: {len(cache.reranked_results)}")
    
    # 设置融合检索完成标志
    return {
        "retrieval_cache": cache,
        "fusion_retrieval_done": True
    }

def context_aggregation_node(state: AgentState) -> dict:
    """上下文聚合节点 - 核心逻辑：去重、累积、增量统计"""
    # 检查两路检索是否都完成
    fusion_done = state.get("fusion_retrieval_done", False)
    sql_done = state.get("sql_retrieval_done", False)
    
    logger.info(f"[context_aggregation_node] 检索状态 - 融合检索: {fusion_done}, SQL检索: {sql_done}")
    
    # 如果两路都还没完成，返回空（等待下一次触发）
    if not (fusion_done and sql_done):
        logger.info(f"[context_aggregation_node] 等待检索完成，跳过本次聚合")
        return {}  # 返回空，不更新任何状态
    
    logger.info(f"[context_aggregation_node] 两路检索均完成，开始聚合")
    cache = state["retrieval_cache"]
    logger.info(f"[context_aggregation_node] 开始上下文聚合，SQL结果: {cache.sql_results is not None}, 重排结果数: {len(cache.reranked_results)}")
    
    # 1. 获取去重集合和累积上下文
    context_id_set = state.get("context_id_set", set())
    accumulated = state.get("accumulated_context", [])
    
    # 2. 收集本轮新增的上下文（去重）
    new_contexts = []
    
    # 2.1 添加 SQL 结果
    if cache.sql_results:
        sql_chunk = ContextChunk(
            id="sql_structured_result",
            content=cache.sql_results,
            source_type="local",
            source_id="sql_database",
            title="结构化数据查询结果",
            similarity=1.0,
            recency=1.0,
            reliability=0.95,
            metadata={"doctype": "sql", "retrieval_source": "sql"},
        )
        if sql_chunk.id not in context_id_set:
            new_contexts.append(sql_chunk)
            context_id_set.add(sql_chunk.id)
            logger.info(f"[context_aggregation_node] 添加SQL结果")
    
    # 2.2 添加重排结果（去重）
    for chunk in cache.reranked_results:
        if chunk.id not in context_id_set:
            new_contexts.append(chunk)
            context_id_set.add(chunk.id)
    logger.info(f"[context_aggregation_node] 添加重排结果，去重后新增 {len(new_contexts)} 个")
    
    # 3. 累积到全局上下文
    accumulated.extend(new_contexts)
    
    # 4. 计算信息增量
    new_gain = len(new_contexts)
    info_gain_log = state.get("info_gain_log", [])
    info_gain_log.append(new_gain)
    
    # 5. 更新连续无增量计数
    consecutive_no_gain = state.get("consecutive_no_gain", 0)
    if new_gain == 0:
        consecutive_no_gain += 1
    else:
        consecutive_no_gain = 0
    
    # 6. 更新查询历史中的结果统计
    current_hop = state.get("current_hop", 0)
    query_history = state.get("query_history", [])
    if query_history and len(query_history) > 0 and query_history[-1].hop == current_hop:
        # 更新当前跳的查询记录
        query_history[-1].result_count = len(cache.reranked_results)
        query_history[-1].new_context_count = new_gain
    
    logger.info(f"[context_aggregation_node] 聚合完成，累积上下文: {len(accumulated)}, 本轮增量: {new_gain}, 连续无增量: {consecutive_no_gain}")
    
    return {
        "accumulated_context": accumulated,
        "context_id_set": context_id_set,
        "info_gain_log": info_gain_log,
        "consecutive_no_gain": consecutive_no_gain,
        "query_history": query_history
    }

def termination_check_node(state: AgentState) -> dict:
    """终止条件检查 - 三重判断机制"""
    from .types import TerminationInfo
    
    termination = TerminationInfo(
        should_terminate=False,
        reason="",
        details=""
    )
    
    # 1. 检查最大跳数
    if state["current_hop"] >= state["max_hops"]:
        termination.should_terminate = True
        termination.reason = "max_hops"
        termination.details = f"已达到最大跳数 {state['max_hops']}"
        logger.info(f"[termination_check_node] 终止检查: {termination.details}")
        return {"termination": termination}
    
    # 2. 检查连续无增量（最高优先级）
    if state["consecutive_no_gain"] >= 2:
        termination.should_terminate = True
        termination.reason = "no_gain"
        termination.details = f"连续 {state['consecutive_no_gain']} 轮未获得新信息"
        logger.info(f"[termination_check_node] 终止检查: {termination.details}")
        return {"termination": termination}
    
    # 3. 检查信息增量趋势（递减检测）
    gain_log = state["info_gain_log"]
    if len(gain_log) >= 3:
        recent_gains = gain_log[-3:]
        total_gain = sum(recent_gains)
        if total_gain <= 1:  # 最近3轮总共只增加 ≤1 个
            termination.should_terminate = True
            termination.reason = "diminishing"
            termination.details = f"信息增量递减，最近3轮仅增加 {total_gain} 个上下文"
            logger.info(f"[termination_check_node] 终止检查: {termination.details}")
            return {"termination": termination}
    
    # logger.info(f"[termination_check_node] 未达到终止条件，继续迭代")
    
    # 重置检索完成标志，为下一轮做准备
    return {
        "termination": termination,
        "fusion_retrieval_done": False,
        "sql_retrieval_done": False
    }

def reasoning_analyzer_node(state: AgentState) -> dict:
    """推理分析器节点"""
    logger.info(f"[reasoning_analyzer_node] 开始推理分析，累积上下文数: {len(state['accumulated_context'])}, 当前跳: {state['current_hop']}/{state['max_hops']}")
    
    # 使用LLM推理分析器进行信息充足性判断
    result = reasoning_analyzer.analyze(
        question=state["original_question"],
        chunks=state["accumulated_context"]
    )
    
    logger.info(f"[reasoning_analyzer_node] 推理分析完成")
    logger.info(f"[reasoning_analyzer_node] 信息充足={result.information_sufficient}, 需要追问={result.need_followup}, 需要联网搜索={result.web_search_needed}")
    
    # 输出结构化缺口信息
    if result.missing_info:
        logger.info(f"[reasoning_analyzer_node] 检测到 {len(result.missing_info)} 个缺口:")
        for i, gap in enumerate(result.missing_info, 1):
            logger.info(f"  缺口 {i}: [类型={gap.type}, 优先级={gap.priority}] {gap.description}")
            logger.info(f"    建议查询: {gap.suggested_query}")
    
    return {
        "information_sufficient": result.information_sufficient,
        "need_followup": result.need_followup,
        "web_search_needed": result.web_search_needed,
        "missing_info": result.missing_info
    }

def followup_query_generation_node(state: AgentState) -> dict:
    """追问生成节点"""
    logger.info(f"[followup_query_generation_node] 开始生成追问，当前问题: {state['current_query']}")
    logger.info(f"[followup_query_generation_node] 缺失信息: {state['missing_info']}")
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
    if result.hallucination_detected and result.reasoning:
        # 如果发现幻觉，输出完整的推理原因
        logger.info(f"[hallucination_detection_node] 检测置信度: {result.confidence}")
        logger.info(f"[hallucination_detection_node] 幻觉原因: {result.reasoning}")
        if result.hallucinated_content:
            logger.info(f"[hallucination_detection_node] 幻觉内容: {result.hallucinated_content}")
    else:
        # 未发现幻觉或无推理信息
        logger.info(f"[hallucination_detection_node] 检测置信度: {result.confidence}")
    
    return {
        "hallucination_detected": result.hallucination_detected,
        "hallucination_confidence": result.confidence,
        "hallucination_reasoning": result.reasoning,
        "hallucinated_content": result.hallucinated_content
    }

def web_search_node(state: AgentState) -> dict:
    """联网搜索节点"""
    logger.info(f"[web_search_node] 开始联网搜索，问题: {state['current_query']}")
    
    # 执行联网搜索
    web_results = retrieval_manager.web_search(state["current_query"], top_k=5)
    logger.info(f"[web_search_node] 联网搜索完成，找到 {len(web_results)} 个结果")
    
    # 转换为ContextChunk格式
    chunks = retrieval_manager.local_retriever._to_chunks(web_results)
    
    # 添加到累积上下文（去重）
    context_id_set = state.get("context_id_set", set())
    accumulated_context = state.get("accumulated_context", [])
    new_count = 0
    
    for chunk in chunks:
        if chunk.id not in context_id_set:
            accumulated_context.append(chunk)
            context_id_set.add(chunk.id)
            new_count += 1
    
    logger.info(f"[web_search_node] 网络搜索结果已添加到累积上下文，新增: {new_count}")
    
    return {
        "accumulated_context": accumulated_context,
        "context_id_set": context_id_set
    }

def final_output_node(state: AgentState) -> dict:
    """最终输出节点"""
    logger.info(f"[final_output_node] 开始生成最终答案，原始问题: {state['original_question']}")
    # 生成最终答案
    final_answer = answer_generator.generate(state["original_question"], state["accumulated_context"])
    logger.info(f"[final_output_node] 最终答案生成完成")
    return {"final_answer": final_answer}

# ================= 条件路由实现 =================

def reasoning_conditional(state: AgentState) -> str:
    """推理分析器条件路由 - 智能决策"""
    
    # 1. 优先检查终止条件
    if state["termination"].should_terminate:
        logger.info(f"[reasoning_conditional] 终止条件触发: {state['termination'].reason}")
        return "answer_generation"  # 强制生成答案
    
    # 2. 信息充足，直接生成
    if state.get("information_sufficient", False):
        logger.info(f"[reasoning_conditional] 信息充足，进入答案生成")
        return "answer_generation"
    
    # 3. LLM 明确判断需要联网搜索
    if state.get("web_search_needed", False):
        logger.info(f"[reasoning_conditional] LLM判断需要联网搜索，执行联网搜索")
        return "web_search"
    
    # 4. 需要追问（有明确缺口或 LLM 判断需要追问）
    if state.get("need_followup", False) or len(state.get("missing_info", [])) > 0:
        logger.info(f"[reasoning_conditional] 需要追问以获取更多信息")
        return "followup_query_generation"
    
    # 5. 兜底：信息不足但不需要联网搜索，直接生成答案（避免无限循环）
    logger.info(f"[reasoning_conditional] 信息不足但无明确后续动作，直接生成答案")
    return "answer_generation"

def hallucination_conditional(state: AgentState) -> str:
    """幻觉检测条件路由"""
    # 1. 优先检查终止条件 - 如果已触发终止，直接输出最终答案（跳过幻觉检测的追问）
    if state["termination"].should_terminate:
        logger.info(f"[hallucination_conditional] 检测到终止条件({state['termination'].reason})，强制输出答案，跳过幻觉检测的追问流程")
        return "final_output"
    
    # 2. 正常流程：检查是否有幻觉
    if state["hallucination_detected"]:
        logger.info(f"[hallucination_conditional] 检测到幻觉，进入追问流程")
        return "followup_query_generation"
    else:
        logger.info(f"[hallucination_conditional] 未检测到幻觉，输出最终答案")
        return "final_output"

def context_availability_conditional(state: AgentState) -> str:
    """上下文可用性条件路由 - 检查是否有足够的上下文继续处理"""
    cache = state["retrieval_cache"]
    accumulated = state.get("accumulated_context", [])
    
    # 检查是否有SQL结果或融合结果
    has_sql = state["sql_retrieval_done"]
    has_reranked = state["fusion_retrieval_done"]
    has_accumulated = len(accumulated) > 0
    
    logger.info(f"[context_availability_conditional] SQL结果: {has_sql}, 重排结果: {has_reranked}, 累积上下文: {has_accumulated}")
    
    # 如果有任何可用的上下文，继续处理
    if has_sql and has_reranked:
        logger.info(f"[context_availability_conditional] 有可用上下文，继续处理")
        return "continue"
    else:
        return "end"
