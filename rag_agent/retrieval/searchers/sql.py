"""
SQL 路由器 - 基于 SQLite 的结构化数据查询
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional


from rag_agent.config import get_config
from rag_agent.llm import llm_services
from ..types import SQLResult
from ...utils.logging import get_logger


class SQLRouter:
    """
    SQL 查询路由器

    使用LLM检测查询是否适合 SQL 查询，并生成相应的 SQL 语句
    """
    _instance: Optional["SQLRouter"] = None
    _initialized: bool = False

    def __new__(cls, db_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        if not SQLRouter._initialized:
            config = get_config()
            self.db_path = db_path or config.sql_db_path
            self.logger = get_logger("SQLRouter")
            try:
                # Use unified LLM service
                self._model = llm_services.get_model()
                self.logger.info("SQLRouter LLM initialized successfully.")
                SQLRouter._initialized = True
            except Exception as e:
                self.logger.warning("SQLRouter LLM init failed; will use fallback. (%s)", e)
                self._model = None
                SQLRouter._initialized = True

    def should_route_to_sql(self, query: str) -> bool:
        """
        使用LLM判断查询是否应该路由到 SQL
        """
        if not self._model:
            # LLM不可用时，使用简单规则作为回退
            query = query.lower()
            # 检查是否包含数值查询模式或财务相关关键词
            has_numeric_or_financial = any(keyword in query for keyword in 
                                          ['是多少', '多少', '同比', '环比', '增长', '下降', '营业收入', '净利润', 'ROE', 'ROA'])
            return has_numeric_or_financial
        
        try:
            prompt = f"""
你是一位专业的查询路由分类专家，能够准确判断用户查询是否需要从结构化数据库中获取信息。

【用户查询】
{query}

【判断标准】
请根据以下标准判断是否需要使用SQL数据库查询：

**需要SQL查询的情况**：
- 涉及具体的数值、财务指标（如营业收入、净利润、ROE、ROA等）
- 需要统计分析、数据汇总、排序比较
- 查询特定时间段的量化数据
- 需要同比、环比分析

**不需要SQL查询的情况**：
- 咨询概念解释、定义、原理等描述性内容
- 询问业务流程、策略分析等定性信息
- 寻求建议、观点或专业意见

【输出要求】
请只输出'是'或'否'，不要添加任何其他解释。
            """
            
            messages = [
                {"role": "system", "content": "你是一位专业的查询路由分类专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._model.invoke(messages)
            result = response.content.strip().lower()
            
            self.logger.debug(f"SQL路由判断结果: {result} (查询: {query})")
            return result == '是'
        except Exception as e:
            self.logger.error(f"SQL路由判断失败: {e}")
            # 异常情况下使用简单规则作为回退
            return any(keyword in query.lower() for keyword in 
                      ['是多少', '多少', '同比', '环比', '增长', '下降', '营业收入', '净利润', 'ROE', 'ROA'])



    def _generate_sql_from_query(self, query: str) -> str:
        """
        使用 LLM 从自然语言查询生成 SQL 语句
        """
        if not self._model:
            self.logger.warning("LLM 模型未初始化，使用默认 SQL 模板")
            return "SELECT metric_name, metric_value, unit, stock_code, company_name, report_period FROM financial_metrics LIMIT 10"

        # 简化的表结构信息
        table_info = """
        表名: financial_metrics
        字段:
        - stock_code (TEXT): 股票代码，如 "600036"
        - company_name (TEXT): 公司名称，如 "招商银行"
        - report_period (TEXT): 报告期，格式如 "2023-Q3", "2023-H1", "2023-FY"
        - metric_name (TEXT): 财务指标名称，如 "营业收入", "净利润", "ROE"
        - metric_value (REAL): 财务指标数值
        - unit (TEXT): 单位，如 "元", "%"
        """

        # 简化的 Text-to-SQL 提示词
        prompt = f"""
你是一位专业的SQL生成专家，擅长将自然语言问题转化为精准的SQL查询语句。

【数据表结构】
{table_info}

【用户查询】
{query}

【SQL生成要求】
请根据用户查询生成对应的SQL语句，遵循以下规则：

1. **使用标准表名**：查询必须使用 `financial_metrics` 表
2. **精准条件**：只根据用户明确提到的条件生成WHERE子句，不要自行添加额外过滤
3. **支持复杂查询**：合理使用 ORDER BY, SUM(), AVG(), GROUP BY 等聚合和排序操作
4. **模糊匹配**：财务指标名称使用 LIKE 进行模糊匹配，支持中英文缩写（如"ROE"和"净资产收益率"）
5. **纯净输出**：只输出SQLSQL语句，不要添加任何解释或标记

【SQL输出】
        """

        try:
            messages = [
                {"role": "system", "content": "你是一位专业的SQL生成专家。"},
                {"role": "user", "content": prompt}
            ]
            response = self._model.invoke(messages)
            sql = response.content.strip()
            
            # 清理 SQL 语句，去除可能的标记
            sql = re.sub(r"^```sql|```$", "", sql).strip()
            self.logger.debug(f"生成的 SQL: {sql}")
            return sql
        except Exception as e:
            self.logger.error(f"SQL 生成失败: {e}")
            return ""

    def execute_query(self, query: str) -> List[SQLResult]:
        """
        执行 SQL 查询（使用 Text-to-SQL）

        Args:
            query: 用户自然语言查询

        Returns:
            SQL 查询结果列表
        """
        if not os.path.exists(self.db_path):
            self.logger.warning(f"SQL 数据库不存在: {self.db_path}")
            return []

        try:
            # 生成 SQL 语句
            sql = self._generate_sql_from_query(query)
            self.logger.info(f"生成的 SQL 语句: {sql}")
            if not sql:
                self.logger.warning("SQL 生成失败，返回空结果")
                return []

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使用字典形式的结果
            cursor = conn.cursor()

            # 执行 SQL
            self.logger.info(f"开始执行 SQL 查询...")
            cursor.execute(sql)
            rows = cursor.fetchall()
            self.logger.info(f"SQL 执行完成，返回 {len(rows)} 行数据")

            results = []
            for row in rows:
                # 使用字典转换来安全访问字段
                row_dict = dict(row)
                
                # 处理聚合查询的情况
                metric_value = 0.0
                if "metric_value" in row_dict:
                    metric_value = row_dict.get("metric_value", 0.0)
                else:
                    # 检查常见的聚合字段名（包含大小写变体）
                    aggregate_fields = ["total_net_profit", "sum_metric_value", "sum(metric_value)", "SUM(metric_value)", "total", "sum"]
                    for field in aggregate_fields:
                        if field in row_dict:
                            metric_value = row_dict.get(field, 0.0)
                            break
                
                result = SQLResult(
                    metric_name=row_dict.get("metric_name", "").strip() if row_dict.get("metric_name") else "",
                    metric_value=metric_value,
                    unit=row_dict.get("unit", "").strip() if row_dict.get("unit") else "",
                    stock_code=row_dict.get("stock_code", "").strip() if row_dict.get("stock_code") else "",
                    company_name=row_dict.get("company_name", "").strip() if row_dict.get("company_name") else "",
                    report_period=row_dict.get("report_period", "").strip() if row_dict.get("report_period") else "",
                    source_table_id=row_dict.get("source_table_id", "").strip() if row_dict.get("source_table_id") else "",
                )
                results.append(result)

            # 去重
            seen = set()
            unique_results = []
            for r in results:
                key = (r.stock_code, r.report_period, r.metric_name, r.metric_value)
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)

            conn.close()
            self.logger.info(f"SQL 查询返回 {len(unique_results)} 条结果（去重后）")
            if unique_results:
                self.logger.info(f"第一条结果示例: {unique_results[0]}")
            return unique_results

        except Exception as e:
            self.logger.error(f"SQL 查询执行失败: {e}")
            return []

    def format_results_as_context(self, results: List[SQLResult]) -> str:
        """将 SQL 结果格式化为上下文文本"""
        if not results:
            return ""

        lines = ["【结构化数据查询结果】"]
        for r in results:
            lines.append(
                f"- {r.company_name}({r.stock_code}) {r.report_period} {r.metric_name}: "
                f"{r.metric_value:,.2f} {r.unit}"
            )

        return "\n".join(lines)