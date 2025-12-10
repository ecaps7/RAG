from typing import TypedDict, List, Optional, Set, Annotated
from operator import add
from .types import ContextChunk, Answer, QueryRecord, MissingInfo, TerminationInfo, RetrievalCache

# 自定义合并函数：合并 RetrievalCache
def merge_retrieval_cache(left: RetrievalCache, right: RetrievalCache) -> RetrievalCache:
    """合并两个 RetrievalCache 对象"""
    return RetrievalCache(
        sql_results=right.sql_results if right.sql_results is not None else left.sql_results,
        vector_results=right.vector_results if right.vector_results else left.vector_results,
        bm25_results=right.bm25_results if right.bm25_results else left.bm25_results,
        fused_results=right.fused_results if right.fused_results else left.fused_results,
        reranked_results=right.reranked_results if right.reranked_results else left.reranked_results,
    )

class AgentState(TypedDict):
    """RAG Agent 状态 - 支持多跳推理与智能终止"""
    
    # ========== 1. 问题与迭代控制 ==========
    original_question: str              # 用户原始问题（不变）
    current_query: str                  # 当前查询（可能是追问）
    current_hop: int                    # 当前跳数 (0-based)
    max_hops: int                       # 最大跳数限制
    
    # ========== 2. 增量记忆（核心优化） ==========
    accumulated_context: List[ContextChunk]     # 累积的所有上下文
    context_id_set: Set[str]                    # 已存在的 context ID（去重）
    entity_history: Set[str]                    # 已搜索的实体/关键词
    query_history: List[QueryRecord]            # 查询历史记录
    
    # ========== 3. 缺口分析结果 ==========
    missing_info: List[MissingInfo]             # 缺失信息列表
    information_sufficient: bool                # 信息是否充足
    need_followup: bool                         # 是否需要追问
    web_search_needed: bool                     # 是否需要联网搜索
    
    # ========== 4. 终止控制 ==========
    info_gain_log: List[int]                    # 每跳的信息增量 [5, 3, 1, 0]
    consecutive_no_gain: int                    # 连续无增量轮次
    termination: TerminationInfo                # 终止判断结果
    
    # ========== 5. 检索缓存（单次迭代的中间结果） ==========
    retrieval_cache: Annotated[RetrievalCache, merge_retrieval_cache]  # 当前迭代的检索结果（支持并行更新）
    
    # ========== 5.1 检索完成标志 ==========
    fusion_retrieval_done: Annotated[bool, lambda x, y: y]  # 融合检索路径是否完成（vector+bm25+rerank）
    sql_retrieval_done: Annotated[bool, lambda x, y: y]      # SQL检索路径是否完成
    
    # ========== 6. 生成与评估 ==========
    generation: Optional[str]                   # 生成的答案文本
    hallucination_detected: bool                # 幻觉检测结果
    
    # ========== 7. 最终输出 ==========
    final_answer: Optional[Answer]              # 最终答案对象