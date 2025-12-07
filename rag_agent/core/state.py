from typing import TypedDict, List, Annotated
import operator
from .types import ContextChunk  #

class AgentState(TypedDict):
    """LangGraph 的核心状态定义"""
    question: str                       # 当前处理的问题（可能是重写后的）
    original_question: str              # 原始用户问题
    documents: List[ContextChunk]       # 检索到的文档片段
    generation: str                     # 生成的答案文本
    web_search_needed: bool             # 是否需要联网搜索
    retry_count: int                    # 重试计数（防止死循环）