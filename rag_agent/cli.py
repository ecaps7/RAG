from __future__ import annotations

import argparse
import json
import os
import csv

import warnings
import logging

# Suppress external library warnings before importing pipeline
try:
    warnings.filterwarnings("ignore", category=UserWarning, module="jieba._compat")
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
except Exception:
    pass
try:
    logging.getLogger("jieba").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    try:
        import jieba  # type: ignore
        jieba.setLogLevel(logging.ERROR)
    except Exception:
        pass
except Exception:
    pass

from typing import List
from .agent import RagAgent
from .memory import rewrite_question
from .utils.debug import set_debug_mode, is_debug_enabled


def warmup_models():
    """预热模型：预先加载 Embedding 和 Cross-encoder 模型到缓存。
    
    这样在用户输入第一个问题时就可以直接使用缓存，无需等待模型加载。
    """
    import time
    from .config import get_config
    
    cfg = get_config()
    debug = is_debug_enabled()
    
    if debug:
        print("\n🔥 预热模型中...")
    
    start_total = time.time()
    
    # 1. 预热 Embedding 模型 + Vector Store
    try:
        from .retrieval.local.vectorstore import get_or_create_vector_store
        if debug:
            print("  ⏳ 加载 Embedding 模型和 Vector Store...")
        t0 = time.time()
        get_or_create_vector_store()
        if debug:
            print(f"  ✅ Vector Store 就绪 (took {time.time() - t0:.2f}s)")
    except Exception as e:
        if debug:
            print(f"  ⚠️ Vector Store 加载失败: {e}")
    
    # 2. 预热 Cross-encoder 模型
    try:
        from .retrieval.reranker import get_or_create_cross_encoder
        model_name = getattr(cfg, "cross_encoder_model", "BAAI/bge-reranker-v2-m3")
        backend = getattr(cfg, "reranker_backend", "ollama")
        
        if getattr(cfg, "use_cross_encoder", True) and backend == "cross_encoder":
            if debug:
                print(f"  ⏳ 加载 Cross-encoder: {model_name}...")
            t0 = time.time()
            get_or_create_cross_encoder(model_name)
            if debug:
                print(f"  ✅ Cross-encoder 就绪 (took {time.time() - t0:.2f}s)")
        elif getattr(cfg, "use_cross_encoder", True) and backend == "ollama" and debug:
             print(f"  ℹ️ 使用 Ollama Reranker ({getattr(cfg, 'ollama_reranker_model', 'bge-m3:567m')})，跳过本地模型加载")
             
    except Exception as e:
        if debug:
            print(f"  ⚠️ Cross-encoder 加载失败: {e}")
    
    # 3. 预热 BM25 索引
    try:
        from .retrieval.local.bm25 import get_or_create_bm25_index
        if debug:
            print("  ⏳ 加载 BM25 索引...")
        t0 = time.time()
        get_or_create_bm25_index()
        if debug:
            print(f"  ✅ BM25 索引就绪 (took {time.time() - t0:.2f}s)")
    except Exception as e:
        if debug:
            print(f"  ⚠️ BM25 索引加载失败: {e}")
    
    total_time = time.time() - start_total
    if debug:
        print(f"🚀 预热完成，总耗时 {total_time:.2f}s\n")


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
        set_debug_mode(True)

    agent = RagAgent(trace_id=args.trace_id)

    # 批量模式优先：从文件读取问题，逐一生成答案并写入（JSONL/CSV 自动识别）
    if args.input:
        in_path = os.path.abspath(args.input)
        out_path = os.path.abspath(args.output)
        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        if not os.path.exists(in_path):
            print(f"[ERROR] 输入文件不存在：{in_path}")
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
        print(f"[DONE] 处理完成，共 {total} 条；成功 {ok}，失败 {err}。输出文件：{out_path}")
        return

    # 单次运行（流式输出最终答案）
    if args.question:
        # 单次运行也预热，这样第一个问题就能快速响应
        warmup_models()
        stream, citations = agent.run_stream(args.question)
        print("=== Final Answer ===")
        for delta in stream:
            print(delta, end="", flush=True)
        print()  # ensure newline after stream
        print("\n=== Citations ===")
        for c in citations:
            print(f"- {c}")
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

                stream, citations = agent.run_stream(rewritten)
                print("=== Final Answer ===")
                final_text_parts: List[str] = []
                for delta in stream:
                    final_text_parts.append(delta)
                    print(delta, end="", flush=True)
                final_text = "".join(final_text_parts)
                print()
                print("\n=== Citations ===")
                if citations:
                    for c in citations:
                        print(f"- {c}")
                else:
                    print("(no citations)")
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

                stream, citations = agent.run_stream(question)
                print("=== Final Answer ===")
                for delta in stream:
                    print(delta, end="", flush=True)
                print()
                print("\n=== Citations ===")
                if citations:
                    for c in citations:
                        print(f"- {c}")
                else:
                    print("(no citations)")
                print()
    except Exception as e:
        print(f"发生错误：{e}")


if __name__ == "__main__":
    main()