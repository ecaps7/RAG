"""FastAPI Web Server for RAG Agent

提供Web前端与RAG Agent的交互接口
- SSE流式问答接口
- 健康检查接口
- 静态文件托管
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import warnings
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 抑制警告
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="RAG Agent Web API",
    description="RAG智能体Web交互接口",
    version="1.0.0"
)

# 配置CORS - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源,生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局RagAgent实例
_agent = None


def get_agent():
    """获取或初始化RagAgent实例"""
    global _agent
    if _agent is None:
        from rag_agent.agent import RagAgent
        _agent = RagAgent()
    return _agent


# ============ 数据模型 ============

class ChatRequest(BaseModel):
    """问答请求模型"""
    question: str
    session_id: str | None = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    service: str


# ============ API路由 ============

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口
    
    Returns:
        服务状态和版本信息
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="RAG Agent"
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式问答接口
    
    使用Server-Sent Events (SSE)推送答案
    
    Args:
        request: 包含用户问题的请求对象
        
    Returns:
        SSE流式响应,包含文本增量和引用信息
    """
    # 验证输入
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    if len(request.question) > 2000:
        raise HTTPException(status_code=400, detail="问题长度不能超过2000字符")
    
    question = request.question.strip()
    logger.info(f"收到问题: {question[:50]}...")
    
    async def generate_sse() -> AsyncIterator[str]:
        """生成SSE事件流"""
        try:
            agent = get_agent()
            
            # 调用agent.run_stream获取流式迭代器和引用列表
            stream_iterator, citation_infos = agent.run_stream(question)
            
            # 推送文本增量
            for delta in stream_iterator:
                data = {
                    "type": "text",
                    "content": delta
                }
                yield f"event: delta\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)  # 让出控制权
            
            # 推送引用信息
            citations = []
            for citation_info in citation_infos:
                citations.append({
                    "ref": citation_info.ref,
                    "title": citation_info.title,
                    "source_id": citation_info.source_id,
                    "source_type": citation_info.source_type,
                    "doc_type": citation_info.doc_type,
                    "page": citation_info.page,
                    "score": citation_info.score,
                    "reliability": citation_info.reliability,
                })
            
            citation_data = {
                "type": "citation",
                "citations": citations
            }
            yield f"event: citation\ndata: {json.dumps(citation_data, ensure_ascii=False)}\n\n"
            
            # 推送完成事件
            done_data = {"type": "done"}
            yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"生成答案时出错: {e}", exc_info=True)
            error_data = {
                "type": "error",
                "message": f"生成答案时出错: {str(e)}"
            }
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
        }
    )


# ============ 静态文件托管 ============

# 挂载静态文件目录(前端HTML/CSS/JS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ============ 启动入口 ============

if __name__ == "__main__":
    import uvicorn
    import sys
    
    print("="*50, file=sys.stderr)
    print("Starting RAG Agent Web Server", file=sys.stderr)
    print("="*50, file=sys.stderr)
    
    # 预热模型
    logger.info("预热RAG Agent模型...")
    print("预热RAG Agent模型...", file=sys.stderr)
    try:
        from rag_agent.cli import warmup_models
        warmup_models()
        logger.info("模型预热完成")
        print("模型预热完成", file=sys.stderr)
    except Exception as e:
        logger.warning(f"模型预热失败: {e}")
        print(f"模型预热失败: {e}", file=sys.stderr)
    
    # 启动服务
    logger.info("启动Web服务...")
    print("启动Web服务于 http://127.0.0.1:8000", file=sys.stderr)
    print("="*50, file=sys.stderr)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
