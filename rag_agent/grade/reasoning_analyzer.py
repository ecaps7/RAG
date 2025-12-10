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
你是一位资深的RAG系统推理分析专家，擅长评估检索到的上下文是否足以完整回答用户问题。

【用户问题】
{question}

【检索到的上下文块】
{chunks}

【分析任务】
请对上述用户问题和检索到的上下文进行全面评估，并做出四个关键判断：

1. **信息是否充足** (information_sufficient)
2. **是否需要追问** (need_followup)
3. **是否需要联网搜索** (web_search_needed)
4. **缺失信息简述** (missing_info_summary) - 仅在 need_followup=true 或 web_search_needed=true 时用1-2句话说明还需要什么信息

【判断指南】

**信息充足的标准** (information_sufficient = true):
- 上下文包含了回答问题所需的**所有关键信息**
- 信息准确、相关且全面，覆盖了问题的各个方面
- 可以直接基于现有上下文生成**可靠且完整**的答案

**需要追问的情况** (need_followup = true):
- 上下文包含了一些相关信息，但**缺少具体细节或关键数据**
- **部分问题有答案，但另一部分缺失**（如对比两家公司，只有一家的数据）
- 问题表述模糊，需要进一步明确具体需求
- 需要更多**内部知识库背景上下文**才能给出完整答案
- 上下文中有相关信息，但需要**更深入挖掘内部文档**
- **数据不完整或不全面**，但可以通过进一步检索内部数据补充

**需要联网搜索的情况** (web_search_needed = true)（**非常重要，请严格判断**）:
- 上下文中**完全没有任何**相关信息（不是数据不完整，而是完全无关）
- 问题需要**实时最新信息**，而上下文中的信息过时（如：最新新闻、实时股价、当前时事）
- 问题需要**外部公共知识或百科信息**，超出现有知识库范围（如：通用百科、历史事件、科学常识）
- 问题明确涉及**互联网公开信息**，且内部知识库无法覆盖
- **注意**：如果问题涉及内部数据（如公司财报、内部文档），即使数据不完整，也应该通过追问获取，而不是联网搜索

**缺失信息简述** (missing_info_summary):
当 need_followup=true 或 web_search_needed=true 时，用1-2句话简要说明还需要查询什么信息。
例如："缺少中信银行的房地产业不良率数据和风险管控策略信息。"

**关键区分**：
- 如果上下文中有相关信息，只是不够详细或数据不完整 → **need_followup = true, web_search_needed = false**
- 如果问题涉及内部数据（财报、公司信息等），即使部分缺失 → **need_followup = true, web_search_needed = false**
- 如果上下文完全没有相关信息，且需要外部公共知识 → **need_followup = false, web_search_needed = true**
- 如果上下文充足 → **information_sufficient = true, need_followup = false, web_search_needed = false, missing_info_summary = ""**

**特别提醒**：
- **不要轻易判断 need web_search**，只有确定需要外部实时信息或公开百科知识时才设置为 true
- **对于内部数据缺失（如财报数据不完整、公司信息缺失），必须优先使用 need_followup**
- **优先考虑 need_followup**，即从内部知识库获取更多信息
- 如果上下文中有任何相关内容（即使只是部分相关），就**不应该**设置 web_search_needed = true
- 判断标准：问题是否需要**外部公开数据源**（如百科、新闻）？如果不需要，就用 need_followup
- **当判断需要追问或联网搜索时，必须简要说明缺失什么信息**
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