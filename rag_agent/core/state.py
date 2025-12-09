from typing import TypedDict, List, Optional
from .types import ContextChunk, Answer

class AgentState(TypedDict):
    """LangGraph 状态定义 - 支持并行检索和多跳"""
    # 基本信息
    question: str                       # 当前处理的问题
    original_question: str              # 原始用户问题
    retry_count: int                    # 重试计数
    max_retries: int                    # 最大重试次数
    
    # 并行检索结果
    sql_results: Optional[str]          # SQL查询结果上下文
    vector_results: List[ContextChunk]  # 向量检索结果
    bm25_results: List[ContextChunk]    # BM25检索结果
    
    # 后处理结果
    fused_results: List[ContextChunk]   # RRF融合结果
    reranked_results: List[ContextChunk] # 重排结果
    final_documents: List[ContextChunk] # 最终上下文
    documents: List[ContextChunk]       # 兼容旧字段
    
    # 生成与评估
    generation: str                     # 生成的答案文本
    hallucination_detected: bool        # 是否检测到幻觉
    
    # 多跳与迭代相关
    followup_queries: List[str]         # 生成的追问列表
    accumulated_context: List[ContextChunk] # 累积的上下文
    
    # 最终输出
    final_answer: Optional[Answer]      # 最终答案
    
    # 控制标志
    web_search_needed: bool             # 是否需要联网搜索
    information_sufficient: bool        # 信息是否充足
    need_followup: bool                 # 是否需要多跳
    
    # 历史记录
    query_history: List[str]            # 查询历史