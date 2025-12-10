"""LLM-based reasoning analyzer implementation."""

from __future__ import annotations

import json
from typing import List, Dict, Optional

from ..core.types import ContextChunk, MissingInfo
from ..llm import llm_services
from ..utils.logging import get_logger
from ..utils.tracing import traceable_step
from .base import BaseReasoningAnalyzer, ReasoningResult
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class ReasoningAnalysisOutput(BaseModel):
    """Output schema for reasoning analysis."""
    information_sufficient: bool = Field(description="信息是否充足以回答问题")
    need_followup: bool = Field(description="是否需要追问内部知识库")
    web_search_needed: bool = Field(description="是否需要联网搜索")
    missing_info_summary: str = Field(
        default="",
        description="缺失信息的简要描述，仅在need_followup=true或web_search_needed=true时填写。用1-2句话说明还需要查询什么信息。"
    )


class LLMReasoningAnalyzer(BaseReasoningAnalyzer):
    """LLM-based reasoning analyzer that assesses information sufficiency."""
    
    def __init__(self, trace_id: Optional[str] = None):
        """Initialize the LLM-based reasoning analyzer.
        
        Args:
            trace_id: Optional trace ID for logging
        """
        self.logger = get_logger(self.__class__.__name__, trace_id)
        # Try to use structured output, with parser as fallback
        self._base_model = llm_services.get_model()
        try:
            self._model = self._base_model.with_structured_output(ReasoningAnalysisOutput)
            self._use_structured = True
            self.logger.info("Using native structured output for reasoning analysis")
        except Exception as e:
            self.logger.warning(f"Native structured output not available, using parser fallback: {e}")
            self._model = self._base_model
            self._use_structured = False
        # Always prepare parser as fallback
        self._parser = PydanticOutputParser(pydantic_object=ReasoningAnalysisOutput)
    
    def _format_chunk_for_analysis(self, chunk: ContextChunk) -> Dict[str, str]:
        """Format a context chunk for analysis."""
        # Get source type from metadata if available
        source_type = chunk.metadata.get('retrieval_source', 'unknown')
        if chunk.source_id == 'sql_database':
            source_type = 'sql'
        
        # Truncate content if too long (keep first 1000 chars)
        content = chunk.content[:1000] + ('...' if len(chunk.content) > 1000 else '')
        
        return {
            "id": chunk.id,
            "content": content,
            "source_type": source_type,
            "title": chunk.title or "",
            "relevance_score": round(chunk.similarity, 3)
        }
    
    def _generate_analysis_prompt(self, question: str, chunks: List[ContextChunk], use_structured: bool = True) -> str:
        """Generate the analysis prompt for the LLM."""
        formatted_chunks = [self._format_chunk_for_analysis(chunk) for chunk in chunks]
        
        prompt = """
你是一位资深的RAG系统推理分析专家，擅长评估检索到的上下文是否足以回答用户问题。你的目标是**务实且高效**地判断信息充足性。

【用户问题】
{question}

【检索到的上下文块】
{chunks}

【分析任务】
请对上述用户问题和检索到的上下文进行评估，并做出四个关键判断：

1. **信息是否充足** (information_sufficient)
2. **是否需要追问** (need_followup)
3. **是否需要联网搜索** (web_search_needed)
4. **缺失信息简述** (missing_info_summary) - 仅在 need_followup=true 或 web_search_needed=true 时用1-2句话说明还需要什么信息

【判断指南】

**信息充足的标准** (information_sufficient = true) - **请采用务实的判断标准**：
- 上下文包含了回答问题的**核心信息和主要论据**
- 能够基于现有上下文生成**有价值且合理**的答案，即使不是100%完美
- 信息的**质量和相关性**足以满足用户的主要需求
- **即使缺少一些次要细节**，但核心问题可以得到解答
- 对于开放性问题，有足够的信息提供**有见地的分析或解释**
- 对于事实性问题，关键数据点和论据已经具备

**重要原则**：
- **80%原则**：如果上下文能覆盖问题的主要方面（80%左右），就应判断为信息充足
- **避免过度追求完美**：不要因为缺少边缘性信息就判定为不充足
- **用户体验优先**：能给出有价值的答案就是充足的，而不是必须面面俱到

**需要追问的情况** (need_followup = true) - **仅在关键信息缺失时**：
- **核心问题无法回答**：缺少回答问题所必需的关键数据或论据
- **多维度问题部分缺失**：如对比两家公司，只有一家的数据，**且另一家数据对回答至关重要**
- **信息明显不完整**：上下文只有碎片化信息，无法形成连贯的答案
- **内部数据缺失**：问题明确需要内部知识库的特定数据，但未检索到
- **注意**：如果只是缺少补充性细节或案例，而核心问题可以回答，则**不需要追问**

**需要联网搜索的情况** (web_search_needed = true) - **谨慎判断，仅限以下场景**：
- 上下文**完全没有相关信息**，且问题涉及外部公开知识（非内部数据）
- 问题明确需要**实时最新信息**（如：当前新闻、实时数据、最新政策）
- 问题需要**通用百科知识或公共常识**，超出内部知识库范围
- 问题涉及**外部实体或事件**，内部知识库不应该包含这些信息
- **严格限制**：如果问题可能涉及内部数据，即使上下文为空，也应优先使用 need_followup 而非 web_search

**缺失信息简述** (missing_info_summary):
当 need_followup=true 或 web_search_needed=true 时，用1-2句话简要说明还需要查询什么**关键信息**。
例如："需要获取中信银行的房地产业不良率数据和风险管控措施。"

**决策流程**（按优先级判断）：
1. **首先评估信息充足性**：上下文能否支持生成有价值的答案？如果能，直接设置 information_sufficient = true
2. **如果信息不足，判断缺失类型**：
   - 缺少的是**内部数据或文档信息** → need_followup = true, web_search_needed = false
   - 缺少的是**外部公开知识或实时信息** → need_followup = false, web_search_needed = true
3. **如果上下文为空**：
   - 问题看起来涉及内部业务数据 → need_followup = true
   - 问题明确需要外部公开信息 → web_search_needed = true

**特别提醒**：
- **降低信息充足的门槛**：能回答核心问题即可，不要追求完美答案
- **减少不必要的追问**：避免因为次要信息缺失而触发追问流程
- **优先内部检索**：不确定时优先使用 need_followup 而非 web_search
- **提升用户体验**：让用户更快得到答案，而不是陷入多轮追问循环
        """
        
        return prompt.format(
            question=question,
            chunks=json.dumps(formatted_chunks, ensure_ascii=False, indent=2)
        )
    
    @traceable_step("reasoning_analysis", run_type="llm")
    def analyze(self, question: str, chunks: List[ContextChunk]) -> ReasoningResult:
        """Analyze the sufficiency of information to answer the question using LLM.
        
        Args:
            question: The user question
            chunks: List of retrieved context chunks
            
        Returns:
            Reasoning result indicating information sufficiency, need for follow-up, etc.
        """
        try:
            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(question, chunks, self._use_structured)
            
            # Get LLM response
            messages = [
                {"role": "system", "content": "你是一位资深的RAG系统推理分析专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = None
            
            # Try structured output first
            if self._use_structured:
                try:
                    response = self._model.invoke(messages)
                    self.logger.info(f"[DEBUG] Structured output successful: {response}")
                except Exception as e:
                    self.logger.warning(f"Structured output failed, falling back to parser: {e}")
                    self._use_structured = False
            
            # Fallback to parser if structured output failed or not available
            if not self._use_structured or response is None:
                llm_response = self._base_model.invoke(messages)
                self.logger.info(f"[DEBUG] Raw LLM response (first 500 chars): {llm_response.content[:500] if hasattr(llm_response, 'content') else str(llm_response)[:500]}")
                response = self._parser.parse(llm_response.content if hasattr(llm_response, 'content') else str(llm_response))
            
            # Convert simple string summary to MissingInfo list if needed
            missing_info_list = []
            if response.missing_info_summary:
                missing_info_list = [
                    MissingInfo(
                        type="general",
                        description=response.missing_info_summary,
                        priority=5,
                        suggested_query=response.missing_info_summary
                    )
                ]
            
            # Map to ReasoningResult object
            return ReasoningResult(
                information_sufficient=response.information_sufficient,
                need_followup=response.need_followup,
                web_search_needed=response.web_search_needed,
                missing_info=missing_info_list
            )
            
        except Exception as e:
            self.logger.error(f"Reasoning analysis failed: {e}")
            self.logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
            self.logger.error(f"[DEBUG] Exception details: {str(e)}")
            import traceback
            self.logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
            # Return default result in case of error
            return ReasoningResult(
                information_sufficient=len(chunks) > 0,
                need_followup=len(chunks) == 0,
                web_search_needed=False,
                missing_info=[]
            )