#!/usr/bin/env python3
"""
测试 SQLRouter 的 Text-to-SQL 功能
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_agent.retrieval.searchers.sql import SQLRouter
from rag_agent.retrieval.config import SQL_DB_PATH


def test_sql_router():
    """测试 SQLRouter 的 Text-to-SQL 功能"""
    print("测试 SQLRouter 的 Text-to-SQL 功能")
    print(f"数据库路径: {SQL_DB_PATH}")
    print("=" * 50)

    # 创建 SQLRouter 实例
    router = SQLRouter()

    # 测试查询列表
    test_queries = [
        "招商银行2025年第一季度的营业收入是多少？",
        "招商银行的净利润是多少？",
        "所有银行的ROE按降序排列",
        "2025年招商银行第一季度的净利润总和是多少？"
    ]

    for i, query in enumerate(test_queries):
        print(f"\n测试查询 {i+1}: {query}")
        print("-" * 50)

        # 判断是否应该路由到 SQL
        should_route = router.should_route_to_sql(query)
        print(f"是否应该路由到 SQL: {should_route}")

        if should_route:
            # 执行查询
            results = router.execute_query(query)
            print(f"查询结果数量: {len(results)}")

            # 打印结果
            if results:
                print("查询结果:")
                for j, result in enumerate(results[:5]):  # 只显示前5个结果
                    # 处理聚合查询的显示
                    if result.company_name and result.stock_code and result.report_period and result.metric_name:
                        print(f"  {j+1}. {result.company_name}({result.stock_code}) {result.report_period} {result.metric_name}: {result.metric_value} {result.unit}")
                    elif result.company_name and result.report_period:
                        print(f"  {j+1}. {result.company_name} {result.report_period} {result.metric_name}: {result.metric_value} {result.unit}")
                    else:
                        # 聚合查询结果的显示
                        print(f"  {j+1}. 查询结果: {result.metric_value} {result.unit}")
                if len(results) > 5:
                    print(f"  ... 还有 {len(results) - 5} 个结果")
            else:
                print("没有查询到结果")


if __name__ == "__main__":
    test_sql_router()