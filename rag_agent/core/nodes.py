"""LangGraph 节点实现"""

from typing import List, Optional
from rag_agent.core.state import AgentState
from rag_agent.core.types import ContextChunk, Answer
from rag_agent.retrieval.retrieval_manager import RetrievalManager
from rag_agent.retrieval.engine import LocalRetriever
from rag_agent.generation.generator import AnswerGenerator
from rag_agent.grade.llm_grader import LLMBasedGrader
from rag_agent.retrieval.types import SearchResult

# 初始化组件
retrieval_manager = RetrievalManager()
local_retriever = LocalRetriever()
answer_generator = AnswerGenerator()
llm_grader = LLMBasedGrader()

# ================= 节点实现 =================

def query_rewrite_node(state: AgentState) -> dict:
    """查询重写/分解节点"""
    # TODO: 实现查询重写逻辑
    # 当前直接返回原始问题
    question = state["original_question"]
    query_history = state.get("query_history", [])
    query_history.append(question)
    return {
        "question": question,
        "query_history": query_history
    }

def sql_generation_execution_node(state: AgentState) -> dict:
    """SQL生成与执行节点 - 可并行"""
    if retrieval_manager.should_route_to_sql(state["question"]):
        sql_results = retrieval_manager.execute_sql_query(state["question"])
    else:
        sql_results = None
    return {"sql_results": sql_results}

def vector_retrieval_node(state: AgentState) -> dict:
    """向量检索节点 - 可并行"""
    vector_results = retrieval_manager.vector_retrieve(state["question"], top_k=10)
    # 转换为ContextChunk格式
    chunks = local_retriever._to_chunks(vector_results)
    return {"vector_results": chunks}

def bm25_retrieval_node(state: AgentState) -> dict:
    """BM25检索节点 - 可并行"""
    bm25_results = retrieval_manager.bm25_retrieve(state["question"], top_k=10)
    # 转换为ContextChunk格式
    chunks = local_retriever._to_chunks(bm25_results)
    return {"bm25_results": chunks}

def rrf_fusion_node(state: AgentState) -> dict:
    """RRF融合节点"""
    result_lists = []
    
    # 准备向量检索结果
    if state["vector_results"]:
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
        fused_search_results = retrieval_manager.rrf_fusion(result_lists)
        chunks = local_retriever._to_chunks(fused_search_results)
        fused_results = chunks
    else:
        fused_results = []
    return {"fused_results": fused_results}

def rerank_node(state: AgentState) -> dict:
    """重排节点"""
    if state["fused_results"]:
        # 转换为SearchResult格式
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
        reranked_search_results = retrieval_manager.rerank(state["question"], search_results, top_k=8)
        chunks = local_retriever._to_chunks(reranked_search_results)
        reranked_results = chunks
    else:
        reranked_results = []
    return {"reranked_results": reranked_results}

def context_aggregation_node(state: AgentState) -> dict:
    """上下文聚合节点 - 处理并行检索结果"""
    final_documents = []
    
    # 添加SQL结果（如果有）
    if state["sql_results"]:
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
    final_documents.extend(state["reranked_results"])
    
    # 累积上下文
    accumulated_context = state.get("accumulated_context", [])
    accumulated_context.extend(final_documents)
    
    return {
        "final_documents": final_documents,
        "documents": final_documents,  # 兼容旧字段
        "accumulated_context": accumulated_context
    }

def reasoning_analyzer_node(state: AgentState) -> dict:
    """推理分析器节点"""
    # 基于上下文数量和相关性判断
    if len(state["final_documents"]) > 0:
        # 简单判断：如果有相关文档且重试次数未达上限，认为信息充足
        information_sufficient = True
        need_followup = False
        web_search_needed = False
        reasoning_result = "信息充足"
    elif state["retry_count"] >= state["max_retries"]:
        # 达到最大重试次数，需要联网搜索
        information_sufficient = False
        need_followup = False
        web_search_needed = True
        reasoning_result = "信息不足，需要联网搜索"
    else:
        # 需要生成追问
        information_sufficient = False
        need_followup = True
        web_search_needed = False
        reasoning_result = "信息不足，需要多跳追问"
    
    return {
        "information_sufficient": information_sufficient,
        "need_followup": need_followup,
        "web_search_needed": web_search_needed,
        "reasoning_result": reasoning_result
    }

def followup_query_generation_node(state: AgentState) -> dict:
    """追问生成节点"""
    # TODO: 实现追问生成逻辑
    # 当前简化实现，实际应调用LLM生成追问
    
    # 生成简单的追问示例
    followup_query = f"关于{state['original_question']}的更多详细信息"
    followup_queries = state.get("followup_queries", [])
    followup_queries.append(followup_query)
    
    return {
        "followup_queries": followup_queries,
        "question": followup_query,
        "retry_count": state["retry_count"] + 1
    }

def answer_generation_node(state: AgentState) -> dict:
    """答案生成节点"""
    answer = answer_generator.generate(state["original_question"], state["accumulated_context"])
    return {"generation": answer.text}

def hallucination_detection_node(state: AgentState) -> dict:
    """幻觉检测节点"""
    # TODO: 实现幻觉检测逻辑
    # 当前简化实现，默认通过
    return {"hallucination_detected": False}

def web_search_node(state: AgentState) -> dict:
    """联网搜索节点"""
    # TODO: 实现联网搜索逻辑
    # 当前简化实现，直接返回
    return {}

def final_output_node(state: AgentState) -> dict:
    """最终输出节点"""
    # 生成最终答案
    final_answer = answer_generator.generate(state["original_question"], state["accumulated_context"])
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

def followup_conditional(state: AgentState) -> str:
    """追问条件路由"""
    return "query_rewrite"

def hallucination_conditional(state: AgentState) -> str:
    """幻觉检测条件路由"""
    if state["hallucination_detected"]:
        return "followup_query_generation"
    else:
        return "final_output"

def web_search_conditional(state: AgentState) -> str:
    """联网搜索条件路由"""
    return "answer_generation"
