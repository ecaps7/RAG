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
        original_question = state["original_question"]
        query_history = state.get("query_history", [])
        
        # 调用LLM生成重写后的查询
        rewrite_prompt = self._build_rewrite_prompt(original_question, query_history)
        rewritten_query = self.generator.generate(rewrite_prompt, "")
        
        # 更新查询历史
        query_history.append(rewritten_query.text)
        
        return {
            "question": rewritten_query.text,
            "query_history": query_history
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
    
    def _build_rewrite_prompt(self, original_question: str, query_history: List[str]) -> str:
        """构建查询重写提示"""
        history_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(query_history)]) if query_history else "无"
        
        return f"""
你是一个专业的查询重写助手，需要将用户的原始问题重写为更适合检索系统的查询。

原始问题：{original_question}
查询历史：
{history_text}

请根据以下要求重写查询：
1. 保持原始问题的核心意图
2. 添加相关的同义词和术语
3. 优化查询结构，提高检索效果
4. 输出简洁明了的重写后的查询，不要添加任何解释

重写后的查询：
"""
    
    def _build_decompose_prompt(self, question: str) -> str:
        """构建查询分解提示"""
        return f"""
你是一个专业的查询分解助手，需要将复杂的用户问题分解为多个简单的子查询。

复杂问题：{question}

请根据以下要求分解查询：
1. 识别问题中的多个子任务或子问题
2. 每个子查询应该独立、具体
3. 子查询数量应根据问题复杂度合理确定
4. 输出格式为每行一个子查询，使用数字序号标记

分解后的子查询：
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
