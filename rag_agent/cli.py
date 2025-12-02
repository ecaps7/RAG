from __future__ import annotations

import argparse
import json
import os
import csv

import warnings
import logging

# 在导入 pipeline 之前抑制外部库告警，以避免模块导入阶段就输出 warning
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
        jieba.setLogLevel(logging.ERROR)  # suppress verbose build logs
    except Exception:
        pass
except Exception:
    pass

from typing import List
from .pipeline import RagAgent
from .memory.short_term import rewrite_question


def main():
    # 已在模块导入前设置告警抑制与外部日志级别，这里无需重复
    parser = argparse.ArgumentParser(description="Run RAG Agent pipeline")
    # 问题参数改为可选：有则单次运行，无则进入交互模式
    parser.add_argument("question", type=str, nargs="?", help="User question")
    parser.add_argument("--trace-id", type=str, default=None, help="Optional trace id for logging")
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
        stream, citations = agent.run_stream(args.question)
        print("=== Final Answer ===")
        for delta in stream:
            print(delta, end="", flush=True)
        print()  # ensure newline after stream
        print("\n=== Citations ===")
        for c in citations:
            print(f"- {c}")
        return

    # 交互式 REPL
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