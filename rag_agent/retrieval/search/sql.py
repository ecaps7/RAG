"""
SQL 路由器 - 基于 SQLite 的结构化数据查询
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from ..config import SQL_DB_PATH
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

    def should_route_to_sql(self, query: str) -> bool:
        """
        判断查询是否应该路由到 SQL

        条件:
        1. 包含数值查询模式
        2. 包含财务指标关键词
        """
        query = query.lower()

        # 检查是否包含数值查询模式
        has_numeric_intent = any(
            re.search(p, query) for p in self.NUMERIC_PATTERNS
        )

        # 检查是否包含财务指标
        has_metric = self._extract_metrics(query) != []

        return has_numeric_intent and has_metric

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

    def execute_query(self, query: str) -> List[SQLResult]:
        """
        执行 SQL 查询（支持 TRIM 和模糊匹配）

        Args:
            query: 用户自然语言查询

        Returns:
            SQL 查询结果列表
        """
        if not os.path.exists(self.db_path):
            self.logger.warning(f"SQL 数据库不存在: {self.db_path}")
            return []

        metrics = self._extract_metrics(query)
        period = self._extract_period(query)
        company = self._extract_company(query)

        if not metrics:
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            results = []

            for metric in metrics:
                # 首先尝试精确匹配（使用 TRIM 去除空格）
                sql = """
                    SELECT metric_name, metric_value, unit, stock_code, company_name, report_period, source_table_id 
                    FROM financial_metrics 
                    WHERE TRIM(metric_name) = TRIM(?)
                """
                params: List[Any] = [metric]

                if period:
                    sql += " AND TRIM(report_period) = TRIM(?)"
                    params.append(period)

                if company:
                    # 股票代码支持模糊匹配（可能带有前缀或后缀）
                    sql += " AND (TRIM(stock_code) = TRIM(?) OR stock_code LIKE ?)"
                    params.append(company)
                    params.append(f"%{company}%")

                cursor.execute(sql, params)
                rows = cursor.fetchall()

                # 如果精确匹配没有结果，尝试模糊匹配
                if not rows:
                    sql_fuzzy = """
                        SELECT metric_name, metric_value, unit, stock_code, company_name, report_period, source_table_id 
                        FROM financial_metrics 
                        WHERE metric_name LIKE ?
                    """
                    params_fuzzy: List[Any] = [f"%{metric}%"]

                    if period:
                        # 报告期也支持模糊匹配
                        period_pattern = period.replace("-", "").replace("_", "")
                        sql_fuzzy += " AND (REPLACE(REPLACE(report_period, '-', ''), '_', '') LIKE ? OR report_period LIKE ?)"
                        params_fuzzy.append(f"%{period_pattern}%")
                        params_fuzzy.append(f"%{period}%")

                    if company:
                        sql_fuzzy += " AND (stock_code LIKE ? OR company_name LIKE ?)"
                        params_fuzzy.append(f"%{company}%")
                        params_fuzzy.append(f"%{company}%")

                    cursor.execute(sql_fuzzy, params_fuzzy)
                    rows = cursor.fetchall()

                    if rows:
                        self.logger.info(
                            f"模糊匹配找到 {len(rows)} 条结果 (指标: {metric})"
                        )

                for row in rows:
                    results.append(
                        SQLResult(
                            metric_name=row[0].strip() if row[0] else row[0],
                            metric_value=row[1],
                            unit=row[2].strip() if row[2] else row[2],
                            stock_code=row[3].strip() if row[3] else row[3],
                            company_name=row[4].strip() if row[4] else row[4],
                            report_period=row[5].strip() if row[5] else row[5],
                            source_table_id=row[6],
                        )
                    )

            # 去重（同一指标可能被多次匹配）
            seen = set()
            unique_results = []
            for r in results:
                key = (r.stock_code, r.report_period, r.metric_name, r.metric_value)
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)

            conn.close()
            self.logger.info(f"SQL 查询返回 {len(unique_results)} 条结果（去重后）")
            return unique_results

        except Exception as e:
            self.logger.error(f"SQL 查询失败: {e}")
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