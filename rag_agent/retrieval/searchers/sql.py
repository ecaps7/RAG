"""
SQL 路由器 - 基于 SQLite 的结构化数据查询
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from ..config import SQL_DB_PATH
from rag_agent.config import get_config, init_model_with_config
from ..types import SQLResult
from ...utils.logging import get_logger


def _cn_to_num(cn: str) -> str:
    """中文数字转阿拉伯数字"""
    mapping = {
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
    }
    return mapping.get(cn, cn)


class SQLRouter:
    """
    SQL 查询路由器

    检测查询是否适合 SQL 查询，并生成相应的 SQL 语句
    """

    # 财务指标关键词映射（模糊匹配到精确指标名）
    METRIC_PATTERNS: Dict[str, List[str]] = {
        # 收入类
        "营业收入": ["营业收入", "营收", "收入"],
        "净利息收入": ["净利息收入", "利息收入", "净利息"],
        "非利息净收入": ["非利息净收入", "非息收入", "中间业务收入"],
        "手续费及佣金净收入": ["手续费", "佣金收入", "中收"],
        # 利润类
        "净利润": ["净利润", "归母净利润", "归属于股东的净利润"],
        "归属于母公司股东的净利润": ["归母净利", "归属母公司", "归母"],
        # 每股指标
        "基本每股收益": ["每股收益", "EPS", "基本每股收益"],
        "每股净资产": ["每股净资产", "BPS"],
        # 资产负债类
        "总资产": ["总资产", "资产总额", "资产规模"],
        "总负债": ["总负债", "负债总额"],
        "贷款总额": ["贷款总额", "贷款余额", "各项贷款"],
        "存款总额": ["存款总额", "存款余额", "各项存款"],
        # 盈利能力
        "净资产收益率": ["ROE", "净资产收益率", "权益回报率"],
        "总资产收益率": ["ROA", "总资产收益率", "资产回报率"],
        "净利差": ["净利差", "NIM", "利差"],
        "净息差": ["净息差", "NIM"],
        # 资产质量
        "不良贷款率": ["不良贷款率", "不良率", "NPL"],
        "拨备覆盖率": ["拨备覆盖率", "拨覆率", "拨备"],
        "关注贷款率": ["关注贷款率", "关注类贷款"],
        # 资本充足
        "核心一级资本充足率": ["核心一级资本充足率", "CET1", "核心资本"],
        "一级资本充足率": ["一级资本充足率", "T1"],
        "资本充足率": ["资本充足率", "CAR"],
    }

    # 数值查询模式
    NUMERIC_PATTERNS = [
        r"是多少",
        r"多少",
        r"达到",
        r"为\s*[\d\.]+",
        r"同比",
        r"环比",
        r"增长",
        r"下降",
        r"变化",
    ]

    # 报告期提取模式
    PERIOD_PATTERNS = [
        (
            r"(\d{4})年第?([一二三四1234])季度?",
            lambda m: f"{m.group(1)}-Q{_cn_to_num(m.group(2))}",
        ),
        (r"(\d{4})年?[上]?半年度?", lambda m: f"{m.group(1)}-H1"),
        (
            r"(\d{4})年度?报|(\d{4})年?年度?",
            lambda m: f"{(m.group(1) or m.group(2))}-FY",
        ),
        (r"(\d{4})[年\-]?Q([1234])", lambda m: f"{m.group(1)}-Q{m.group(2)}"),
        (r"(\d{4})[年\-]?H([12])", lambda m: f"{m.group(1)}-H{m.group(2)}"),
    ]

    # 常见银行简称映射
    COMPANY_MAP = {
        "招商银行": "600036",
        "招行": "600036",
        "中信银行": "601998",
        "中信": "601998",
        "平安银行": "000001",
        "平安": "000001",
        "工商银行": "601398",
        "工行": "601398",
        "建设银行": "601939",
        "建行": "601939",
        "农业银行": "601288",
        "农行": "601288",
        "中国银行": "601988",
        "中行": "601988",
        "交通银行": "601328",
        "交行": "601328",
        "兴业银行": "601166",
        "兴业": "601166",
        "浦发银行": "600000",
        "浦发": "600000",
        "民生银行": "600016",
        "民生": "600016",
        "光大银行": "601818",
        "光大": "601818",
        "华夏银行": "600015",
        "华夏": "600015",
    }

    def __init__(self, db_path: str = SQL_DB_PATH):
        self.db_path = db_path
        self.logger = get_logger("SQLRouter")
        try:
            cfg = get_config()
            self._model = init_model_with_config(
                cfg.response_model_name,
                cfg.response_model_temperature
            )
        except Exception as e:
            self.logger.warning("SQLRouter LLM init failed; will use fallback. (%s)", e)
            self._model = None

    def should_route_to_sql(self, query: str) -> bool:
        """
        判断查询是否应该路由到 SQL

        条件:
        1. 包含数值查询模式 或者
        2. 包含财务指标关键词并且包含排序、分组、聚合等操作意图
        """
        query = query.lower()

        # 检查是否包含数值查询模式
        has_numeric_intent = any(
            re.search(p, query) for p in self.NUMERIC_PATTERNS
        )

        # 检查是否包含财务指标
        has_metric = self._extract_metrics(query) != []
        
        # 检查是否包含排序、分组、聚合等操作意图
        has_complex_intent = any(keyword in query for keyword in ['排序', '分组', '总和', '平均', '最大值', '最小值', '降序', '升序'])

        return has_numeric_intent and has_metric or (has_metric and has_complex_intent)

    def _extract_metrics(self, query: str) -> List[str]:
        """从查询中提取财务指标名称"""
        metrics = []
        query_lower = query.lower()

        for metric_name, patterns in self.METRIC_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in query_lower:
                    metrics.append(metric_name)
                    break

        return metrics

    def _extract_period(self, query: str) -> Optional[str]:
        """从查询中提取报告期"""
        for pattern, formatter in self.PERIOD_PATTERNS:
            match = re.search(pattern, query)
            if match:
                return formatter(match)
        return None

    def _extract_company(self, query: str) -> Optional[str]:
        """从查询中提取公司名/股票代码"""
        for name, code in self.COMPANY_MAP.items():
            if name in query:
                return code

        # 提取股票代码（6位数字，可能带后缀如 .SH .SZ）
        code_match = re.search(r"(\d{6})(?:\.(?:SH|SZ|sh|sz))?", query)
        if code_match:
            return code_match.group(1)

        return None

    def _generate_sql_from_query(self, query: str) -> str:
        """
        使用 LLM 从自然语言查询生成 SQL 语句

        Args:
            query: 用户自然语言查询

        Returns:
            生成的 SQL 语句
        """
        if not self._model:
            self.logger.warning("LLM 模型未初始化，使用默认 SQL 模板")
            # 回退到原始的参数提取方式，但使用更安全的默认SQL
            return "SELECT metric_name, metric_value, unit, stock_code, company_name, report_period, source_table_id FROM financial_metrics LIMIT 10"

        # 表结构信息
        table_schema = """
            CREATE TABLE financial_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT,           -- 股票代码，如 "600036"
                company_name TEXT,         -- 公司名称，如 "招商银行"
                report_period TEXT,        -- 报告期，格式如 "2023-Q3", "2023-H1", "2023-FY"
                metric_name TEXT,          -- 财务指标名称，如 "营业收入", "净利润", "ROE"
                metric_value REAL,         -- 财务指标数值，如 10000000000.0
                unit TEXT,                 -- 单位，如 "元", "%"
                source_table_id TEXT       -- 源表ID
            )
        """

        # 示例数据
        sample_data = """
            示例数据：
            1,600036,招商银行,2023-Q3,营业收入,311450000000.0,元,table_1
            2,600036,招商银行,2023-Q3,净利润,108530000000.0,元,table_1
            3,600036,招商银行,2023-Q3,ROE,16.56,% ,table_1
            4,601398,工商银行,2023-Q3,营业收入,712120000000.0,元,table_2
            5,601398,工商银行,2023-Q3,净利润,265000000000.0,元,table_2
        """

        # Text-to-SQL 提示词
        prompt = f"""
        请根据以下信息将用户的自然语言查询转换为准确的 SQL 语句：

        1. 数据库表结构：
        {table_schema}

        2. 示例数据：
        {sample_data}

        3. 注意事项：
        - 只返回 SQL 语句，不要包含其他解释或说明
        - 使用正确的表名 financial_metrics
        - 针对中文查询，确保理解用户意图
        - 支持复杂查询，如 ORDER BY, SUM(), GROUP BY 等
        - 注意数据类型，metric_value 是数值类型
        - 股票代码、公司名称和财务指标名称都可以使用模糊匹配
        - 财务指标名称可能包含详细描述（如"年化加权平均净资产收益率"、"归属于母公司股东的净利润"）
        - 请使用 LIKE '%指标关键词%' 来查询财务指标，确保能匹配到相关的所有指标变体
        - 对于英文缩写指标（如ROE、ROA、EPS），请同时匹配缩写和对应的中文名称（如"净资产收益率"、"总资产收益率"、"每股收益"）
        - 只根据用户查询中明确提到的条件生成SQL，不要添加用户未提及的额外过滤条件（如时间范围、单位等）
        - 如果用户没有指定特定的报告期，请不要添加时间过滤条件
        - 如果用户没有指定特定的单位，请不要添加单位过滤条件

        4. 用户查询：
        {query}
        """

        try:
            messages = [
                {"role": "system", "content": "你是一个专业的 SQL 生成器，能够根据用户的自然语言查询和数据库表结构生成准确的 SQL 语句。"},
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
            if not sql:
                return []

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使用字典形式的结果
            cursor = conn.cursor()

            # 执行 SQL
            cursor.execute(sql)
            rows = cursor.fetchall()

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
            self.logger.debug(f"SQL 查询返回 {len(unique_results)} 条结果（去重后）")
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