from typing import List, Dict, Any
from rag_agent.core.state import AgentState
from rag_agent.generation.generator import AnswerGenerator

class QueryRewriter:
    """查询重写/分解器"""
    
    def __init__(self):
        self.generator = AnswerGenerator()
    
    def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """执行查询重写
        
        Args:
            state: 当前代理状态
            
        Returns:
            包含重写后查询的字典
        """
        from rag_agent.core.types import QueryRecord
        
        current_query = state["current_query"]
        original_question = state["original_question"]
        query_history = state.get("query_history", [])
        current_hop = state.get("current_hop", 0)
        
        # 构建重写 Prompt（只关注当前查询优化，不考虑历史）
        rewrite_prompt = self._build_rewrite_prompt(current_query)
        rewritten_query = self.generator.generate(rewrite_prompt, "")
        
        # 如果是第一跳，初始化查询历史
        if current_hop == 0 and not query_history:
            query_history = [
                QueryRecord(
                    hop=0,
                    query=rewritten_query.text,  # 使用重写后的查询
                    intent="initial",
                    result_count=0,
                    new_context_count=0
                )
            ]
            return {
                "current_query": rewritten_query.text,
                "query_history": query_history
            }
        
        return {
            "current_query": rewritten_query.text
        }
    
    def decompose_query(self, question: str) -> List[str]:
        """分解复杂查询为多个子查询
        
        Args:
            question: 原始问题
            
        Returns:
            子查询列表
        """
        # 调用LLM分解查询
        decompose_prompt = self._build_decompose_prompt(question)
        decomposition = self.generator.generate(decompose_prompt, "")
        
        # 解析子查询列表
        sub_queries = self._parse_sub_queries(decomposition.text)
        
        return sub_queries
    
    def _build_rewrite_prompt(self, current_query: str) -> str:
        """构建查询重写提示
        
        注意：查询重写只关注当前查询本身的优化，不考虑历史查询记录。
        历史信息应该在追问生成或多跳推理决策阶段使用。
        """
        return f"""
你是一位资深的检索系统优化专家，擅长将用户的自然语言问题转化为高效的检索查询。

【当前查询】
{current_query}

【优化要求】
请将上述查询优化为更适合检索系统的查询语句：

1. **保持核心意图**：精准把握查询的真实需求，不偏离问题本质
2. **扩展关键术语**：补充相关的专业术语、同义词、缩写或全称
3. **优化查询结构**：调整词序和表达方式，使其更符合知识库的组织形式
4. **提取关键点**：突出查询中的核心关键词和实体
5. **简洁明确**：直接输出优化后的查询语句，无需任何解释说明

【优化后的查询】
"""
    
    def _build_decompose_prompt(self, question: str) -> str:
        """构建查询分解提示"""
        return f"""
你是一位擅长问题分析的专家，能够将复杂的多层次问题拆解为清晰的子任务。

【复杂问题】
{question}

【分解要求】
请将上述复杂问题分解为多个独立的子查询，以便逐步检索和解答：

1. **识别子任务**：从问题中提取出所有需要分别回答的子问题或子任务
2. **保持独立性**：每个子查询应当独立完整，可以单独执行检索
3. **具体明确**：子查询应具体且目标明确，避免模糊表述
4. **合理数量**：根据问题复杂度确定子查询数量（通常2-5个），避免过度拆分
5. **标准格式**：每行一个子查询，使用数字序号（如1.)标记

【分解后的子查询】
"""
    
    def _parse_sub_queries(self, response: str) -> List[str]:
        """解析生成的子查询列表"""
        sub_queries = []
        lines = response.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                # 提取数字序号后的查询内容
                sub_query = line.split(".", 1)[1].strip()
                if sub_query:
                    sub_queries.append(sub_query)
        
        return sub_queries
