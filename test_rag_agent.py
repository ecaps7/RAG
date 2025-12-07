#!/usr/bin/env python3
"""测试RAG Agent"""

from rag_agent.agent import RagAgent

# 创建RAG Agent实例
agent = RagAgent()

# 测试问题
question = "招商银行2025年第一季度的净利润是多少？"

print(f"测试问题: {question}")
print("=" * 50)

try:
    # 调用run方法
    answer = agent.run(question)
    
    print(f"答案: {answer.text}")
    print(f"置信度: {answer.confidence:.2f}")
    print(f"引用: {answer.citations}")
    print(f"元数据: {answer.meta}")
    
    print("\n测试通过!")
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
