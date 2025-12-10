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
        from rag_agent.core.types import QueryRecord
        
        original_question = state["original_question"]
        accumulated_context = state.get("accumulated_context", [])
        current_hop = state.get("current_hop", 0)
        query_history = state.get("query_history", [])
        missing_info = state.get("missing_info", [])
        
        # 构建上下文总结
        context_summary = self._summarize_context(accumulated_context)
        
        # 如果有结构化的缺口信息，使用其描述作为缺口分析
        if missing_info and len(missing_info) > 0:
            # 取优先级最高的缺口
            top_gap = missing_info[0]
            gap_analysis = top_gap.description
        else:
            # 否则通过LLM分析上下文缺口
            gap_analysis = self._analyze_context_gaps(original_question, context_summary)
        
        # 基于缺口分析生成追问查询
        followup_prompt = self._build_followup_prompt(
            original_question, 
            context_summary, 
            gap_analysis, 
            query_history
        )
        followup_result = self.generator.generate(followup_prompt, "")
        followup_query = followup_result.text
        
        # 更新查询历史
        new_record = QueryRecord(
            hop=current_hop + 1,
            query=followup_query,
            intent="followup",
            result_count=0,
            new_context_count=0
        )
        query_history.append(new_record)
        
        return {
            "current_query": followup_query,
            "query_history": query_history,
            "current_hop": current_hop + 1
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
你是一位信息分析专家，擅长识别用户问题与现有上下文之间的知识缺口。

【用户问题】
{question}

【已有上下文总结】
{context_summary}

【缺口分析任务】
请仔细对比用户问题和已有上下文，找出还缺少哪些关键信息才能完整回答该问题。

请以条目列表形式输出缺失的信息点，每行一个，使用数字序号标记：
"""
        
        gap_result = self.generator.generate(gap_prompt, "")
        return gap_result.text
    
    def _build_followup_prompt(self, question: str, context_summary: str, 
                              gap_analysis: str, query_history: List) -> str:
        """构建追问生成提示"""
        if query_history:
            followup_history_text = "\n".join([
                f"{i+1}. Hop {record.hop}: {record.query}"
                for i, record in enumerate(query_history)
            ])
        else:
            followup_history_text = "无"
        
        return f"""
你是一位智能问答系统的信息补充专家，擅长根据信息缺口生成精准的追问查询。

【用户原始问题】
{question}

【当前已有上下文总结】
{context_summary}

【识别出的信息缺口】
{gap_analysis}

【历史追问记录】
{followup_history_text}

【追问生成要求】
请基于上述信息缺口分析，生成一个适合检索系统的查询语句，用于获取缺失的关键信息：

1. **直接针对缺口**：追问应直接针对已识别出的主要信息缺口，具体明确
2. **提取关键实体和概念**：从缺口描述中提取核心关键词、实体名称和指标
3. **保持相关性**：追问必须与原始问题紧密相关，服务于回答原问题的目标
4. **避免重复**：不要重复之前的追问内容，从新角度补充信息
5. **简洁清晰**：直接输出一个简洁有效的查询语句（而不是描述性文字），无需任何解释说明

【示例】
缺口描述：“缺少中信银行的房地产业不良率数据，无法完整比较两家银行的房地产业不良率。”
生成追问：“中信银行房地产业不良贷款率”

【生成的追问查询】
"""
