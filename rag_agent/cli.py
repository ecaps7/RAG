from __future__ import annotations

import argparse
import json
import os
import csv

import warnings
import logging

# Suppress ALL warnings before importing any libraries
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from .utils.logging import get_logger

# Get logger for CLI
logger = get_logger("RAG_CLI")

# Suppress external library warnings before importing pipeline
try:
    warnings.filterwarnings("ignore", category=UserWarning, module="jieba._compat")
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
    warnings.filterwarnings("ignore", message=".*max_length.*is ignored.*")
except Exception:
    pass
try:
    logging.getLogger("jieba").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    try:
        import jieba  # type: ignore
        jieba.setLogLevel(logging.ERROR)
    except Exception:
        pass
except Exception:
    pass

from typing import List, Set
import re as regex_module
from .agent import RagAgent
# from .memory import rewrite_question
from .utils.logging import set_logging_debug_mode, is_logging_debug_mode
from .core.types import CitationInfo


def extract_cited_refs(text: str) -> Set[int]:
    """从文本中提取 [n] 格式的引用编号。"""
    matches = regex_module.findall(r'\[(\d+)\]', text)
    return {int(m) for m in matches}


def format_citations(
    citation_infos: List[CitationInfo],
    cited_refs: Set[int],
    show_all: bool = False
) -> str:
    """格式化引用输出。
    
    Args:
        citation_infos: 所有引用信息
        cited_refs: LLM 实际引用的编号集合
        show_all: 是否显示所有引用（调试用）
    
    Returns:
        格式化的引用字符串
    """
    lines = []
    lines.append("\n---")
    lines.append("**📊 数据来源 (References)**\n")
    
    cited_count = 0
    uncited_count = 0
    
    for info in citation_infos:
        if info.ref in cited_refs:
            cited_count += 1
            
            # 根据文档类型确定类型标签
            if info.doc_type == "sql":
                type_tag = "[结构化数据]"
            elif info.doc_type == "table":
                type_tag = "[表格]"
            else:
                type_tag = "[文本]"
            
            # 格式化页码信息（仅非 SQL 数据显示）
            if info.doc_type != "sql" and info.page:
                page_tag = f" (Page: {info.page})"
            else:
                page_tag = ""
            
            lines.append(f"* **[{info.ref}]** {type_tag} {info.title}{page_tag}")
            lines.append("")
        else:
            uncited_count += 1
    
    if uncited_count > 0:
        lines.append(f"*(已过滤 {uncited_count} 条未引用的检索源)*")
    
    if cited_count == 0:
        lines.append("*(未检测到引用标记，显示所有检索源)*\n")
        for info in citation_infos:
            # 根据文档类型确定类型标签
            if info.doc_type == "sql":
                type_tag = "[结构化数据]"
            elif info.doc_type == "table":
                type_tag = "[表格]"
            else:
                type_tag = "[文本]"
            page_tag = f" (Page: {info.page})" if info.page and info.doc_type != "sql" else ""
            lines.append(f"* [{info.ref}] {type_tag} {info.title}{page_tag}")
    
    return "\n".join(lines)


def warmup_models():
    """预热模型：预先加载搜索引擎组件、Embedding 模型和 Reranker 模型到缓存。
    
    这样在用户输入第一个问题时就可以直接使用缓存，无需等待模型加载。
    预热内容：
    1. LocalRetriever (Milvus + BM25 + SQL)
    2. Ollama Embedding 模型 (qwen3-embedding:4b)
    3. Reranker 模型 (Qwen3-Reranker-0.6B)
    """
    import time
    
    debug = is_logging_debug_mode()
    
    if debug:
        logger.info("预热模型中...")
    
    start_total = time.time()
    
    # 预热本地混合检索器（包含 Milvus + BM25 + SQL）
    try:
        from .retrieval import get_retriever
        if debug:
            logger.debug("加载 LocalRetriever (Milvus + BM25 + SQL)...")
        t0 = time.time()
        retriever = get_retriever()
        # 触发内部组件初始化
        retriever.vector_searcher._ensure_client()
        retriever.bm25_searcher._ensure_loaded()
        if debug:
            logger.debug(f"LocalRetriever 就绪 (took {time.time() - t0:.2f}s)")
    except Exception as e:
        if debug:
            logger.warning(f"LocalRetriever 加载失败: {e}")
    
    # 预热 Ollama Embedding 模型
    try:
        import ollama
        from .config import get_config
        config = get_config()
        if debug:
            logger.debug(f"预热 Embedding 模型 ({config.ollama_embed_model})...")
        t0 = time.time()
        # 使用一个短文本触发模型加载
        _ = ollama.embeddings(model=config.ollama_embed_model, prompt="预热")
        if debug:
            logger.debug(f"Embedding 模型就绪 (took {time.time() - t0:.2f}s)")
    except Exception as e:
        if debug:
            logger.warning(f"Embedding 模型加载失败: {e}")
    
    # 预热 Reranker 模型（HuggingFace Qwen3-Reranker）
    try:
        from .retrieval.rankers import SemanticReranker
        if debug:
            logger.debug("加载 Reranker 模型 (Qwen3-Reranker-0.6B)...")
        t0 = time.time()
        reranker = SemanticReranker()
        # 触发模型加载
        reranker._load_model()
        if debug:
            logger.debug(f"Reranker 模型就绪 (took {time.time() - t0:.2f}s)")
    except Exception as e:
        if debug:
            logger.warning(f"Reranker 模型加载失败: {e}")
    
    total_time = time.time() - start_total
    if debug:
        logger.info(f"预热完成，总耗时 {total_time:.2f}s")


def main():
    # 已在模块导入前设置告警抑制与外部日志级别，这里无需重复
    parser = argparse.ArgumentParser(description="Run RAG Agent pipeline")
    # 问题参数改为可选：有则单次运行，无则进入交互模式
    parser.add_argument("question", type=str, nargs="?", help="User question")
    parser.add_argument("--trace-id", type=str, default=None, help="Optional trace id for logging")
    parser.add_argument("--debug", action="store_true", help="启用调试模式，显示 pipeline 每层的彩色详细日志")
    # 批量模式：从文件读取问题并写 JSONL 输出
    parser.add_argument("--input", type=str, default=None, help="包含问题的文本文件（每行一个问题）")
    parser.add_argument(
        "--output", type=str, default=os.path.join("outputs", "answers.jsonl"),
        help="批量模式输出文件（JSONL）"
    )
    parser.add_argument("--append", action="store_true", help="批量模式写入时追加到输出文件")
    parser.add_argument("--enable-memory", action="store_true", help="交互模式开启短期记忆（LangGraph）")
    parser.add_argument("--thread-id", type=str, default=None, help="记忆会话ID（默认使用 trace-id 或 'repl'）")
    args = parser.parse_args()

    # 启用调试模式
    if args.debug:
        set_logging_debug_mode(True)

    agent = RagAgent(trace_id=args.trace_id)

    # 批量模式优先：从文件读取问题，逐一生成答案并写入（JSONL/CSV 自动识别）
    if args.input:
        in_path = os.path.abspath(args.input)
        out_path = os.path.abspath(args.output)
        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        if not os.path.exists(in_path):
            logger.error(f"输入文件不存在：{in_path}")
            return

        mode = "a" if args.append and os.path.exists(out_path) else "w"
        total = 0
        ok = 0
        err = 0
        # 根据扩展名选择输出格式
        is_csv = out_path.lower().endswith(".csv")
        with open(in_path, "r", encoding="utf-8") as fin:
            if is_csv:
                # CSV 写入（与 qa_with_refs.csv 一致的三列表头）
                with open(out_path, mode, encoding="utf-8", newline="") as fout:
                    writer = csv.writer(fout)
                    if mode == "w":
                        writer.writerow(["问题", "标准答案", "引用来源"])  # 写表头
                    for line in fin:
                        q = (line or "").strip()
                        if not q:
                            continue
                        total += 1
                        try:
                            ans = agent.run(q)
                            citations_text = "; ".join([str(c) for c in (ans.citations or []) if str(c).strip()])
                            writer.writerow([q, ans.text, citations_text])
                            ok += 1
                        except Exception as e:
                            writer.writerow([q, f"错误：{e}", ""])
                            err += 1
            else:
                # JSONL 写入（默认）
                with open(out_path, mode, encoding="utf-8") as fout:
                    for line in fin:
                        q = (line or "").strip()
                        if not q:
                            continue
                        total += 1
                        try:
                            ans = agent.run(q)
                            row = {
                                "question": q,
                                "answer": ans.text,
                                "citations": ans.citations,
                                "confidence": ans.confidence,
                                "meta": ans.meta,
                            }
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                            ok += 1
                        except Exception as e:
                            row = {"question": q, "error": str(e)}
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                            err += 1
        logger.info(f"处理完成，共 {total} 条；成功 {ok}，失败 {err}。输出文件：{out_path}")
        return

    # 单次运行（流式输出最终答案）
    if args.question:
        # 单次运行也预热，这样第一个问题就能快速响应
        warmup_models()
        stream, citation_infos = agent.run_stream(args.question)
        print("=== Final Answer ===")
        final_text_parts: List[str] = []
        for delta in stream:
            final_text_parts.append(delta)
            print(delta, end="", flush=True)
        final_text = "".join(final_text_parts)
        print()  # ensure newline after stream
        
        # 解析引用并格式化输出
        cited_refs = extract_cited_refs(final_text)
        print(format_citations(citation_infos, cited_refs))
        return

    # 交互式 REPL - 预热模型
    warmup_models()
    print("RAG Agent 交互模式：输入问题，输入 /q 退出。")
    try:
        if args.enable_memory:
            # 默认流式输出 + 短期记忆：使用查询改写驱动检索与生成
            messages: List[dict] = []
            while True:
                try:
                    question = input("问> ").strip()
                except EOFError:
                    print("\n已退出。")
                    break
                except KeyboardInterrupt:
                    print("\n已退出。")
                    break

                if not question:
                    continue
                if question.lower() in {"/q", "q", ":q", "exit", "quit"}:
                    break

                # 追加用户消息，并基于历史进行查询改写
                messages.append({"role": "user", "content": question})
                rewritten = rewrite_question(messages)

                stream, citation_infos = agent.run_stream(rewritten)
                print("=== Final Answer ===")
                final_text_parts: List[str] = []
                for delta in stream:
                    final_text_parts.append(delta)
                    print(delta, end="", flush=True)
                final_text = "".join(final_text_parts)
                print()
                
                # 解析引用并格式化输出
                cited_refs = extract_cited_refs(final_text)
                print(format_citations(citation_infos, cited_refs))
                print()

                # 将助手消息写入记忆，供后续改写使用
                messages.append({"role": "assistant", "content": final_text})
        else:
            while True:
                try:
                    question = input("问> ").strip()
                except EOFError:
                    print("\n已退出。")
                    break
                except KeyboardInterrupt:
                    print("\n已退出。")
                    break

                if not question:
                    continue
                if question.lower() in {"/q", "q", ":q", "exit", "quit"}:
                    break

                stream, citation_infos = agent.run_stream(question)
                print("=== Final Answer ===")
                final_text_parts: List[str] = []
                for delta in stream:
                    final_text_parts.append(delta)
                    print(delta, end="", flush=True)
                final_text = "".join(final_text_parts)
                print()
                
                # 解析引用并格式化输出
                cited_refs = extract_cited_refs(final_text)
                print(format_citations(citation_infos, cited_refs))
                print()
    except Exception as e:
        print(f"发生错误：{e}")


if __name__ == "__main__":
    main()