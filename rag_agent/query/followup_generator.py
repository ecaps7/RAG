from typing import List, Dict, Any
from rag_agent.core.state import AgentState
from rag_agent.core.types import ContextChunk
from rag_agent.generation.generator import AnswerGenerator

class FollowupQueryGenerator:
    """追问查询生成器"""
    
    def __init__(self):
        self.generator = AnswerGenerator()
    
    def generate_followup(self, state: AgentState) -> Dict[str, Any]:
        """生成追问查询
        
        Args:
            state: 当前代理状态
            
        Returns:
            包含追问查询的字典
        """
        original_question = state["original_question"]
        final_documents = state["final_documents"]
        retry_count = state["retry_count"]
        followup_queries = state.get("followup_queries", [])
        
        # 分析上下文，识别信息缺口
        context_summary = self._summarize_context(final_documents)
        gap_analysis = self._analyze_context_gaps(original_question, context_summary)
        
        # 生成追问查询
        followup_prompt = self._build_followup_prompt(
            original_question, 
            context_summary, 
            gap_analysis, 
            followup_queries
        )
        followup_query = self.generator.generate(followup_prompt, "")
        
        # 更新状态
        followup_queries.append(followup_query.text)
        
        return {
            "followup_queries": followup_queries,
            "question": followup_query.text,
            "retry_count": retry_count + 1
        }
    
    def _summarize_context(self, documents: List[ContextChunk]) -> str:
        """总结上下文信息"""
        if not documents:
            return "无相关上下文信息"
        
        context_items = []
        for i, doc in enumerate(documents):
            source_type = doc.source_type or "unknown"
            source_id = doc.source_id or "unknown"
            title = doc.title or "无标题"
            content = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            
            context_item = f"{i+1}. [{source_type}] {title} ({source_id})\n{content}"
            context_items.append(context_item)
        
        return "\n\n".join(context_items)
    
    def _analyze_context_gaps(self, question: str, context_summary: str) -> str:
        """分析上下文信息缺口"""
        # 调用LLM分析信息缺口
        gap_prompt = f"""
请分析以下问题和提供的上下文，识别出上下文中缺少哪些信息来回答问题：

问题：{question}

上下文总结：
{context_summary}

请简要列出缺失的信息点，每行一个，使用数字序号标记：
"""
        
        gap_result = self.generator.generate(gap_prompt, "")
        return gap_result.text
    
    def _build_followup_prompt(self, question: str, context_summary: str, 
                              gap_analysis: str, followup_history: List[str]) -> str:
        """构建追问生成提示"""
        followup_history_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(followup_history)]) if followup_history else "无"
        
        return f"""
你是一个专业的AI助手，需要根据当前上下文和用户的原始问题，生成针对性的追问查询。

原始问题：{question}

当前上下文总结：
{context_summary}

信息缺口分析：
{gap_analysis}

之前的追问历史：
{followup_history_text}

请生成一个针对性的追问查询，用于获取缺失的信息，要求：
1. 追问应该具体、明确，针对识别出的信息缺口
2. 保持与原始问题的相关性
3. 输出简洁明了的追问查询，不要添加任何解释
4. 避免重复之前的追问

追问查询：
"""
