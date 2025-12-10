"""LLM-based hallucination detector implementation."""

from __future__ import annotations

import json
from typing import List, Dict, Optional

from ..core.types import ContextChunk
from ..llm import llm_services
from ..utils.logging import get_logger
from ..utils.tracing import traceable_step
from .base import BaseHallucinationDetector, HallucinationResult
from pydantic import BaseModel, Field


class HallucinationDetectionOutput(BaseModel):
    """Output schema for hallucination detection."""
    hallucination_detected: bool = Field(description="Whether hallucination is detected")
    confidence: float = Field(description="Confidence score of the detection (0.0-1.0)")
    reasoning: Optional[str] = Field(description="Reasoning for the detection, only provided if hallucination is detected")
    hallucinated_content: Optional[str] = Field(description="The hallucinated content if detected")


class LLMHallucinationDetector(BaseHallucinationDetector):
    """LLM-based hallucination detector that checks answer consistency with context."""
    
    def __init__(self, trace_id: Optional[str] = None):
        """Initialize the LLM-based hallucination detector.
        
        Args:
            trace_id: Optional trace ID for logging
        """
        self.logger = get_logger(self.__class__.__name__, trace_id)
        # Use unified LLM service with structured output
        self._model = llm_services.get_structured_model(HallucinationDetectionOutput)
    
    def _format_chunk_for_detection(self, chunk: ContextChunk) -> Dict[str, str]:
        """Format a context chunk for hallucination detection."""
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
    
    def _generate_detection_prompt(self, question: str, answer: str, chunks: List[ContextChunk]) -> str:
        """Generate the hallucination detection prompt for the LLM."""
        formatted_chunks = [self._format_chunk_for_detection(chunk) for chunk in chunks]
        
        prompt = """
你是一位专业的RAG系统幻觉检测专家，擅长识别生成的答案中是否包含未被上下文支持的信息。

【用户问题】
{question}

【生成的答案】
{answer}

【上下文块】
{chunks}

【检测任务】
请仔细比对生成的答案与提供的上下文，执行以下检测步骤：

1. **逐句比对**：将答案中的每一句话与上下文进行比对
2. **识别无据信息**：找出答案中不存在于任何上下文中的信息
3. **判断幻觉性质**：评估这些无据信息是否构成幻觉
4. **提供理由** (如果检测到幻觉)：简要说明哪些信息缺少支持或与上下文矛盾
5. **提取幻觉内容** (如果检测到幻觉)：指出具体的幻觉文本
6. **给出置信度** (0.0-1.0)：对检测结果的确定性评分
7. **无幻觉处理**：如果未发现幻觉，将 reasoning 设置为 None

【幻觉判断标准】

**属于幻觉的情况**：
- 信息在所有上下文中完全不存在
- 信息与上下文明确矛盾或冲突
- 具体的事实、数字、名称或日期在上下文中未提及
- 无根据的断言或主张

**不属于幻觉的情况**：
- 基于上下文的合理推理和归纳
- 对上下文信息的重新表述或转述
- 公认的常识性知识，且不与上下文冲突
        """
        
        return prompt.format(
            question=question,
            answer=answer,
            chunks=json.dumps(formatted_chunks, ensure_ascii=False, indent=2)
        )
    
    @traceable_step("hallucination_detection", run_type="llm")
    def detect(self, question: str, answer: str, chunks: List[ContextChunk]) -> HallucinationResult:
        """Detect hallucinations in the generated answer using LLM.
        
        Args:
            question: The user question
            answer: The generated answer
            chunks: List of context chunks used for generation
            
        Returns:
            Hallucination detection result
        """
        try:
            # Generate detection prompt
            prompt = self._generate_detection_prompt(question, answer, chunks)
            
            # Get LLM response with structured output
            messages = [
                {"role": "system", "content": "你是一位专业的RAG系统幻觉检测专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._model.invoke(messages)
            
            # Map to HallucinationResult object
            return HallucinationResult(
                hallucination_detected=response.hallucination_detected,
                confidence=response.confidence,
                reasoning=response.reasoning,
                hallucinated_content=response.hallucinated_content
            )
            
        except Exception as e:
            self.logger.error(f"Hallucination detection failed: {e}")
            
            # Return default result in case of error
            return HallucinationResult(
                hallucination_detected=False,
                confidence=0.5,
                reasoning=None,
                hallucinated_content=None
            )