"""LangGraph 状态机构建 - 支持并行检索"""

from langgraph.graph import StateGraph, END
from rag_agent.core.state import AgentState
from rag_agent.core.nodes import (
    query_rewrite_node,
    sql_generation_execution_node,
    vector_retrieval_node,
    bm25_retrieval_node,
    rrf_fusion_node,
    rerank_node,
    context_aggregation_node,
    termination_check_node,
    reasoning_analyzer_node,
    followup_query_generation_node,
    answer_generation_node,
    hallucination_detection_node,
    web_search_node,
    final_output_node,
    terminate_conditional,
    reasoning_conditional,
    hallucination_conditional,
    context_availability_conditional
)

# 构建状态机
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("query_rewrite", query_rewrite_node)
graph.add_node("sql_generation_execution", sql_generation_execution_node)
graph.add_node("vector_retrieval", vector_retrieval_node)
graph.add_node("bm25_retrieval", bm25_retrieval_node)
graph.add_node("rrf_fusion", rrf_fusion_node)
graph.add_node("rerank", rerank_node)
graph.add_node("context_aggregation", context_aggregation_node)
graph.add_node("termination_check", termination_check_node)
graph.add_node("reasoning_analyzer", reasoning_analyzer_node)
graph.add_node("followup_query_generation", followup_query_generation_node)
graph.add_node("answer_generation", answer_generation_node)
graph.add_node("hallucination_detection", hallucination_detection_node)
graph.add_node("web_search", web_search_node)
graph.add_node("final_output", final_output_node)

# 设置入口点
graph.set_entry_point("query_rewrite")

# 构建并行检索流程
# 1. 查询重写 -> 向量检索和BM25检索并行执行
graph.add_edge("query_rewrite", "vector_retrieval")
graph.add_edge("query_rewrite", "bm25_retrieval")

# 2. 查询重写 -> SQL生成执行（并行）
graph.add_edge("query_rewrite", "sql_generation_execution")

# 3. 向量检索和BM25检索结果 -> RRF融合
graph.add_edge("vector_retrieval", "rrf_fusion")
graph.add_edge("bm25_retrieval", "rrf_fusion")

# 4. RRF融合 -> 重排
graph.add_edge("rrf_fusion", "rerank")

# 5. 重排 -> 上下文聚合
graph.add_edge("rerank", "context_aggregation")

# 6. SQL生成执行 -> 上下文聚合（与重排结果并行到达，等待所有输入完成）
graph.add_edge("sql_generation_execution", "context_aggregation")

# 7. 上下文聚合 -> 条件路由（检查是否有可用上下文）
graph.add_conditional_edges(
    "context_aggregation",
    context_availability_conditional,
    {
        "continue": "termination_check",
        "end": END
    }
)

# 8. 终止检查 -> 推理分析器
graph.add_conditional_edges(
    "termination_check",
    terminate_conditional,
    {
        "continue": "reasoning_analyzer",
        "answer_generation": "answer_generation"
    }
)

# 9. 推理分析器 -> 条件路由
graph.add_conditional_edges(
    "reasoning_analyzer",
    reasoning_conditional,
    {
        "answer_generation": "answer_generation",
        "web_search": "web_search",
        "followup_query_generation": "followup_query_generation"
    }
)

# 10. 追问生成 -> 查询重写（进入下一轮检索）
graph.add_edge("followup_query_generation", "query_rewrite")

# 11. 答案生成 -> 幻觉检测
graph.add_edge("answer_generation", "hallucination_detection")

# 12. 幻觉检测 -> 条件路由
graph.add_conditional_edges(
    "hallucination_detection",
    hallucination_conditional,
    {
        "final_output": "final_output",
        "followup_query_generation": "followup_query_generation"
    }
)

# 13. 联网搜索 -> 答案生成
graph.add_edge("web_search", "answer_generation")

# 14. 最终输出 -> END
graph.add_edge("final_output", END)

# 编译状态机
app = graph.compile()
