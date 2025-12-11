"""
表格数据处理与入库脚本

功能：
1. 解析标准化表格数据（包含全局文档信息和表格详情）
2. 使用 LLM 从 HTML 表格中提取结构化财务指标
3. 生成向量 Embedding 并存入 Milvus
4. 构建 BM25 关键词索引
5. 将提取的指标存入 SQLite

数据格式：
{
  "document": {
    "source": "CMB-2025-q1",
    "company": "招商银行",
    "company_full": "招商银行股份有限公司",
    "stock_code": "600036.SH / 03968.HK",
    "report_period": "2025年第一季度",
    "report_type": "季报",
    "fiscal_year": "2025"
  },
  "tables": [
    {
      "id": "TABLE_1",
      "summary": "表格摘要...",
      "page": 2,
      "section": ["章节1", "章节2"],
      "raw_html": "<table>...</table>",
      "context": {"before": "...", "after": "..."},
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

使用示例：
    python process_table_ingest.py --input-file outputs/CMB-2025-q1/CMB-2025-q1-table.json
"""

import sqlite3
import json
import jieba
import pickle
import ollama
import numpy as np
import os
import re
from pymilvus import MilvusClient
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field

load_dotenv()

# ================= 配置区域 =================
# 数据库文件路径
SQL_DB_PATH = "database/financial_rag.db"
MILVUS_DB_PATH = "database/financial_vectors.db"
BM25_INDEX_PATH = "database/bm25_index.pkl"

# Embedding 配置
OLLAMA_MODEL = "qwen3-embedding:4b"
EMBEDDING_DIM = 2560  # Qwen3-Embedding 4B 的标准维度

# LLM 配置（用于结构化指标提取）- 豆包模型
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("ARK_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "doubao-seed-1-6-251015")

# 明确排除的表格类型（不包含财务指标）
EXCLUDE_TABLE_KEYWORDS = [
    "股东情况", "股东信息", "董事", "监事", "高管",
    "公司治理", "关联交易", "重大事项", "公司基本情况"
]

# ================= Pydantic 模型定义 =================

class FinancialMetric(BaseModel):
    """单个财务指标"""
    metric_name: str = Field(description="指标名称，如：营业收入、净利润、不良贷款率等")
    metric_value: float = Field(description="指标数值，纯数字（去除千分位逗号）")
    unit: str = Field(description="单位，如：百万元、亿元、元、%等")

class MetricsExtractionResult(BaseModel):
    """财务指标提取结果"""
    metrics: List[FinancialMetric] = Field(default_factory=list, description="提取的财务指标列表")

# ================= LLM 指标提取 Prompt =================

METRIC_EXTRACTION_PROMPT = """你是专业的跨银行多周期财务数据提取助手，能适配不同银行、不同报告期间（季度/半年度/年度）的财务报告表格，精准提取标准化指标。请从以下 HTML 表格中提取关键财务指标：

## 报告基础信息
- 公司名称: {company_name}
- 股票代码: {stock_code}
- 报告期时间范围: {report_period}（例：2025年1-3月、2024年1-12月）
- 表格来源: {table_id}

## HTML 表格内容
```html
{raw_html}
```

## 核心提取规则
1. 仅提取**当期数据**：报告期时间范围内的发生额（如收入、利润）、报告期末的时点额（如资产、负债），不提取同比/环比增减率、上年同期/上年末数据及增减额；
2. 指标三要素完整：每个指标必须包含「标准化指标名称」「纯数字数值」「明确单位」，缺一不可；
3. 数值格式统一：去除千分位逗号，负数以负号（-）表示，保留原始精度（无小数按整数、有小数按原位数）；
4. 单位规范：优先使用表格标注单位（如无标注，参考常见单位：百万元、亿元、元、%、个百分点），避免单位混淆（例：明确区分「%」与「个百分点」）；
5. 报表优先级：优先提取合并报表数据，若表格无合并报表标识或仅为母公司报表，需在指标名称后标注「（母公司）」；
6. 指标名称标准化：统一指标命名（例：「归属于母公司股东的净利润」「核心一级资本充足率（高级法）」，避免「归属于股东净利润」「高级法下核心一级资本充足率」等不统一表述）；
7. 重复指标处理：同一指标在表格中多次出现时，取合并报表数据（无合并报表则取最新出现的有效值），不重复提取。

## 重点提取指标类型及标准化名称
### 1. 收入类
- 营业收入
- 净利息收入
- 非利息净收入
- 手续费及佣金净收入
- 投资收益
- 汇兑净收益（或：汇兑收益，根据表格表述适配）
- 其他净收入小计

### 2. 利润类
- 净利润
- 归属于母公司股东的净利润（或：归属于股东的净利润，根据表格表述适配）
- 扣除非经常性损益后归属于母公司股东的净利润（或：扣非净利润，根据表格表述适配）

### 3. 每股指标
- 基本每股收益
- 稀释每股收益
- 每股净资产（或：归属于普通股股东的每股净资产，根据表格表述适配）

### 4. 资产类
- 资产总额
- 贷款和垫款总额
- 客户存款总额
- 金融投资余额
- 不良贷款余额
- 关注贷款余额
- 逾期贷款余额

### 5. 负债类
- 负债总额

### 6. 权益类
- 股东权益合计（或：股东权益）
- 归属于母公司股东的权益（或：归属于股东权益，根据表格表述适配）

### 7. 盈利能力
- 净资产收益率（ROE/ROAE）（年化数据优先，标注「（年化）」）
- 扣除非经常性损益后净资产收益率（ROE/ROAE）（年化数据优先，标注「（年化）」）
- 总资产收益率（ROA/ROAA）（年化数据优先，标注「（年化）」）
- 净利差
- 净利息收益率

### 8. 资产质量
- 不良贷款率
- 关注贷款率
- 逾期贷款率
- 拨备覆盖率
- 贷款拨备率

### 9. 资本充足
- 核心一级资本充足率（高级法）（如有）
- 核心一级资本充足率（权重法）（如有）
- 一级资本充足率（高级法）（如有）
- 一级资本充足率（权重法）（如有）
- 资本充足率（高级法）（如有）
- 资本充足率（权重法）（如有）

### 10. 现金流
- 经营活动产生的现金流量净额

请将提取的指标以结构化格式返回。如果表格中没有可提取的财务指标，返回空列表。
"""

# ================= 工具函数 =================

llm_client = None

def get_llm_client():
    """获取或初始化 LLM 客户端"""
    global llm_client
    if llm_client is None:
        if not LLM_API_KEY:
            raise ValueError("❌ 未设置 LLM_API_KEY 或 DASHSCOPE_API_KEY 环境变量")
        llm_client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_API_BASE
        )
    return llm_client


def load_table_json(file_path: str) -> Tuple[Dict, List[Dict]]:
    """
    加载标准化表格JSON文件
    
    格式: {document: {...}, tables: [...]}
    
    Returns:
        (document_context, tables) 元组
        - document_context: 文档级全局元数据字典
        - tables: 表格列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict) or 'tables' not in data:
        raise ValueError(f"❌ 无效的JSON格式: {file_path}\n期望格式: {{document: {{...}}, tables: [...]}}")
    
    doc = data.get('document', {})
    tables = data.get('tables', [])
    
    # 构造标准化的 document_context（全局文档信息）
    stock_code = doc.get('stock_code', '')
    normalized_stock_code = stock_code.split('/')[0].strip() if '/' in stock_code else stock_code
    company_short = doc.get('company', '')
    
    # 标准化公司名称
    company_name = _normalize_company_name(company_short, normalized_stock_code)
    
    doc_ctx = {
        'source': doc.get('source', ''),
        'company_name': company_name,  # 使用标准化后的简称
        'company_short': company_short,
        'stock_code': normalized_stock_code,  # 直接使用已规范化的代码
        'report_period': doc.get('report_period', ''),
        'report_type': doc.get('report_type', ''),
        'fiscal_year': doc.get('fiscal_year', ''),
        'data_scope': '集团'
    }
    
    return doc_ctx, tables


def extract_metrics_from_table(table_item: Dict, doc_ctx: Dict) -> List[Dict]:
    """
    使用 LLM 从 HTML 表格中提取结构化财务指标（支持原生结构化输出）
    
    Args:
        table_item: 表格数据项，包含 raw_html, summary, section 等
        doc_ctx: 全局文档上下文信息
        
    Returns:
        提取的指标列表，每项包含 metric_name, metric_value, unit
    """
    try:
        raw_html = table_item.get('raw_html', '')
        if not raw_html or len(raw_html.strip()) < 20:
            return []
        
        # 获取表格元信息
        table_id = table_item.get('id', 'unknown')
        section = table_item.get('section', [])
        section_str = ' > '.join(section) if isinstance(section, list) else str(section)
        
        # 排除明确不包含财务指标的表格
        if any(kw in section_str for kw in EXCLUDE_TABLE_KEYWORDS):
            return []
        
        # 获取全局文档信息
        company_name = doc_ctx.get('company_short', doc_ctx.get('company_name', '未知'))
        stock_code = doc_ctx.get('stock_code', '未知')
        report_period = doc_ctx.get('report_period', '未知')
        
        # 构建 prompt
        prompt = METRIC_EXTRACTION_PROMPT.format(
            company_name=company_name,
            stock_code=stock_code,
            report_period=report_period,
            table_id=table_id,
            raw_html=raw_html
        )
        
        # 调用 LLM（使用原生结构化输出）
        client = get_llm_client()
        
        try:
            # 尝试使用原生结构化输出
            completion = client.beta.chat.completions.parse(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=MetricsExtractionResult,
                temperature=0.1
            )
            
            result = completion.choices[0].message.parsed
            if result and result.metrics:
                metrics = [m.model_dump() for m in result.metrics]
            else:
                metrics = []
                
        except Exception as e:
            # 降级到传统方式
            print(f"   ⚠️ 原生结构化输出失败，降级到传统方式: {e}")
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 解析 JSON
            json_match = re.search(r'\[[\s\S]*?\]', result_text)
            if json_match:
                metrics = json.loads(json_match.group())
            else:
                metrics = []
        
        # 添加来源信息
        normalized_period = _normalize_report_period(report_period)
        # stock_code 和 company_name 已在 doc_ctx 中规范化，直接使用
        
        for m in metrics:
            m['stock_code'] = stock_code  # doc_ctx 中已是规范化后的值
            m['company_name'] = company_name
            m['report_period'] = normalized_period
            m['source_table_id'] = table_id
        
        return metrics
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失败 (表格 {table_item.get('id', 'unknown')}): {e}")
        return []
    except Exception as e:
        print(f"❌ LLM 提取失败 (表格 {table_item.get('id', 'unknown')}): {e}")
        return []


def _normalize_company_name(entity: str, stock_code: str) -> str:
    """
    标准化公司名称，使用统一的简称
    
    Args:
        entity: 实体名称
        stock_code: 股票代码
        
    Returns:
        标准化后的公司名称
    """
    # 根据股票代码映射
    code_to_name = {
        '600036.SH': '招商银行',
        '601998.SH': '中信银行'
    }
    
    if stock_code in code_to_name:
        return code_to_name[stock_code]
    
    # 根据实体名称映射
    name_map = {
        'CMB': '招商银行',
        '招商银行股份有限公司': '招商银行',
        '招商银行': '招商银行',
        '招行': '招商银行',
        'CITIC': '中信银行',
        '中信银行股份有限公司': '中信银行',
        '中信银行': '中信银行',
        '中信': '中信银行'
    }
    
    for key, name in name_map.items():
        if key in (entity or ''):
            return name
    
    return entity or '未知'


def _normalize_report_period(period: str) -> str:
    """
    标准化报告期格式
    例如: "2025年第一季度" -> "2025-Q1"
    """
    period = period.strip()
    
    # 匹配 "2025年第一季度" 或 "二〇二五年第一季度"
    q_match = re.search(r'(二〇\d{2}|20\d{2})年第([一二三四])季度', period)
    if q_match:
        year_str = q_match.group(1)
        # 转换中文年份
        if year_str.startswith('二〇'):
            year = '20' + year_str[2:]
        else:
            year = year_str
        q_map = {'一': 'Q1', '二': 'Q2', '三': 'Q3', '四': 'Q4'}
        quarter = q_map.get(q_match.group(2), 'Q1')
        return f"{year}-{quarter}"
    
    # 匹配 "2025年1-3月"
    month_match = re.search(r'(\d{4})年(\d+)[-~](\d+)月', period)
    if month_match:
        year = month_match.group(1)
        end_month = int(month_match.group(3))
        if end_month == 3:
            return f"{year}-Q1"
        elif end_month == 6:
            return f"{year}-Q2"
        elif end_month == 9:
            return f"{year}-Q3"
        elif end_month == 12:
            return f"{year}-Q4"
    
    # 匹配 "2025年半年度"
    h_match = re.search(r'(\d{4})年(半年度|上半年)', period)
    if h_match:
        year = h_match.group(1)
        return f"{year}-H1"
    
    # 匹配 "2025年年度"
    y_match = re.search(r'(\d{4})年(年度|度)', period)
    if y_match:
        year = y_match.group(1)
        return f"{year}-FY"
    
    return period


def get_embedding(text: str) -> List[float]:
    """调用 Ollama 生成向量"""
    try:
        text = text.replace("\n", " ").strip()
        if not text:
            return np.zeros(EMBEDDING_DIM).tolist()
            
        response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
        embedding = response.get('embedding')
        
        if not embedding or len(embedding) != EMBEDDING_DIM:
            print(f"⚠️ 警告: 向量维度异常或为空，返回零向量")
            return np.zeros(EMBEDDING_DIM).tolist()
            
        return embedding
    except Exception as e:
        print(f"❌ Embedding 调用失败: {e}")
        return np.zeros(EMBEDDING_DIM).tolist()


def init_sqlite() -> sqlite3.Connection:
    """初始化 SQLite 结构化指标库"""
    db_dir = os.path.dirname(SQL_DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"   📁 创建目录: {db_dir}")
    
    conn = sqlite3.connect(SQL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT,
            company_name TEXT,
            report_period TEXT,
            metric_name TEXT,
            metric_value REAL,
            unit TEXT,
            source_table_id TEXT,
            UNIQUE(stock_code, report_period, metric_name)
        )
    ''')
    conn.commit()
    return conn


def init_milvus() -> Tuple[MilvusClient, str]:
    """初始化 Milvus 向量库"""
    db_dir = os.path.dirname(MILVUS_DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"   📁 创建目录: {db_dir}")
    
    client = MilvusClient(uri=MILVUS_DB_PATH)
    collection_name = "financial_chunks"
    
    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=EMBEDDING_DIM,
            metric_type="COSINE",
            auto_id=True
        )
    
    return client, collection_name


# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(
        description="process_table_ingest.py: 处理标准化表格数据并入库"
    )
    parser.add_argument("--input-file", type=str, required=True,
                        help="输入表格数据文件路径（JSON格式，包含 document 和 tables）")
    args = parser.parse_args()
    
    input_file = args.input_file
    
    # 1. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误：文件不存在 {input_file}")
        return
    
    # 2. 初始化数据库
    print("🛠️ 正在初始化数据库...")
    sql_conn = init_sqlite()
    sql_cursor = sql_conn.cursor()
    milvus_client, collection_name = init_milvus()
    
    # 3. 加载表格数据
    print(f"📂 正在读取表格文件: {input_file}")
    doc_ctx, tables = load_table_json(input_file)
    
    print(f"   ✓ 加载了 {len(tables)} 个表格")
    print(f"   ✓ 文档: {doc_ctx.get('company_short', 'Unknown')} - {doc_ctx.get('report_period', 'Unknown')}")
    
    # ---------------------------------------------------------
    # A. SQL Layer: 使用 LLM 从表格中提取结构化指标
    # ---------------------------------------------------------
    print("📊 正在使用 LLM 提取结构化指标...")
    
    all_metrics = []
    tables_processed = 0
    
    for table in tables:
        metrics = extract_metrics_from_table(table, doc_ctx)
        if metrics:
            all_metrics.extend(metrics)
            tables_processed += 1
            print(f"   ✓ {table.get('id', 'unknown')}: 提取了 {len(metrics)} 个指标")
    
    # 去重：同一公司、同一报告期、同一指标名称只保留一条
    # 如果有多个来源提取了相同指标，优先保留第一个（通常是更前面的表格）
    seen = set()
    unique_metrics = []
    for m in all_metrics:
        key = (m['stock_code'], m['report_period'], m['metric_name'])
        if key not in seen:
            seen.add(key)
            unique_metrics.append(m)
        else:
            # 记录被去重的数据（用于调试）
            pass  # 可以在这里添加日志
    
    # 写入 SQLite
    if unique_metrics:
        sql_records = [
            (
                m['stock_code'],
                m['company_name'],
                m['report_period'],
                m['metric_name'],
                m['metric_value'],
                m['unit'],
                m.get('source_table_id', 'unknown')
            )
            for m in unique_metrics
        ]
        
        sql_cursor.executemany('''
            INSERT OR REPLACE INTO financial_metrics 
            (stock_code, company_name, report_period, metric_name, metric_value, unit, source_table_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', sql_records)
        sql_conn.commit()
        print(f"   --> 从 {tables_processed} 个表格中提取并存入 {len(unique_metrics)} 条指标（去重后）")
    else:
        print("   --> 未提取到任何指标")
    
    # ---------------------------------------------------------
    # B. Vector + Keyword Layer: 处理向量和 BM25
    # ---------------------------------------------------------
    print("🧠 正在处理向量 Embedding 和 BM25 分词...")
    
    milvus_data = []
    bm25_corpus = []
    doc_map = []
    
    for table in tables:
        # 使用 summary 作为检索文本
        content = table.get('summary', '')
        if not content or len(content.strip()) < 10:
            continue
        
        # 1. 生成向量
        vec = get_embedding(content)
        
        # 2. 准备 Milvus 数据（包含全局文档信息）
        metadata_dict = {
            "source_id": str(table.get('id', 'unknown')),
            "type": "table",
            "page": str(table.get('page', 0)),
            # 关键：表格存 raw_html，这是给 LLM 看的原始数据
            "raw_data": table.get('raw_html', ''),
            "section": ' > '.join(table.get('section', [])) if isinstance(table.get('section'), list) else str(table.get('section', '')),
            # 全局文档信息（贯穿所有数据层）
            "company_name": doc_ctx.get('company_short', doc_ctx.get('company_name', '')),
            "stock_code": doc_ctx.get('stock_code', ''),
            "report_period": doc_ctx.get('report_period', ''),
            "report_type": doc_ctx.get('report_type', ''),
            "fiscal_year": doc_ctx.get('fiscal_year', ''),
            "source": doc_ctx.get('source', '')
        }
        
        entry = {
            "vector": vec,
            "text": content,  # 存摘要用于语义搜索
            "subject": "table_summary",
            "metadata": json.dumps(metadata_dict, ensure_ascii=False)
        }
        milvus_data.append(entry)
        
        # 3. 准备 BM25 数据
        tokens = list(jieba.cut_for_search(content))
        bm25_corpus.append(tokens)
        doc_map.append(entry)
    
    # ---------------------------------------------------------
    # C. 写入存储
    # ---------------------------------------------------------
    
    # 1. 写入 Milvus（去重）
    if milvus_data:
        print(f"💾 正在写入 Milvus ({len(milvus_data)} 条数据)...")
        
        # 检查已存在的数据 (使用 company_name + source_id 组合去重)
        existing_keys = set()
        try:
            batch_size = 16384
            offset = 0
            while True:
                batch_data = milvus_client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["metadata"],
                    limit=batch_size,
                    offset=offset
                )
                if not batch_data:
                    break
                    
                for item in batch_data:
                    try:
                        metadata = json.loads(item.get('metadata', '{}'))
                        company = metadata.get('company_name', '')
                        source_id = metadata.get('source_id', '')
                        # 使用公司名+source_id组合作为唯一键
                        existing_keys.add(f"{company}::{source_id}")
                    except:
                        pass
                
                if len(batch_data) < batch_size:
                    break
                offset += batch_size
            
            if existing_keys:
                print(f"   ℹ️ 发现 {len(existing_keys)} 条已存在的数据")
        except Exception as e:
            print(f"   ⚠️ 无法检查已存在数据: {e}")
        
        # 过滤重复数据 (基于 company_name + source_id 组合)
        new_data = []
        skipped = 0
        for entry in milvus_data:
            metadata = json.loads(entry['metadata'])
            company = metadata.get('company_name', '')
            source_id = metadata.get('source_id', '')
            key = f"{company}::{source_id}"
            if key not in existing_keys:
                new_data.append(entry)
            else:
                skipped += 1
        
        if new_data:
            res = milvus_client.insert(collection_name=collection_name, data=new_data)
            print(f"   --> 成功写入 {res['insert_count']} 条新向量，跳过 {skipped} 条重复数据")
        else:
            print(f"   --> 所有数据已存在，跳过 {skipped} 条重复数据")
    
    # 2. 构建并保存 BM25（支持追加）
    existing_doc_map = []
    if os.path.exists(BM25_INDEX_PATH):
        try:
            with open(BM25_INDEX_PATH, 'rb') as f:
                _, existing_doc_map = pickle.load(f)
            print(f"   ℹ️ 发现已有 BM25 索引，包含 {len(existing_doc_map)} 条文档")
        except Exception as e:
            print(f"   ⚠️ 读取已有 BM25 索引失败: {e}")
    
    if bm25_corpus or existing_doc_map:
        print("📑 正在构建/更新 BM25 索引...")
        from rank_bm25 import BM25Okapi
        
        # 检查重复 (使用 company_name + source_id 组合)
        existing_keys_bm25 = set()
        for entry in existing_doc_map:
            try:
                metadata = json.loads(entry.get('metadata', '{}'))
                company = metadata.get('company_name', '')
                source_id = metadata.get('source_id', '')
                existing_keys_bm25.add(f"{company}::{source_id}")
            except:
                pass
        
        # 过滤重复数据 (基于 company_name + source_id 组合)
        filtered_corpus = []
        filtered_doc_map = []
        skipped_bm25 = 0
        
        for i, entry in enumerate(doc_map):
            try:
                metadata = json.loads(entry.get('metadata', '{}'))
                company = metadata.get('company_name', '')
                source_id = metadata.get('source_id', '')
                key = f"{company}::{source_id}"
                if key not in existing_keys_bm25:
                    filtered_corpus.append(bm25_corpus[i])
                    filtered_doc_map.append(entry)
                else:
                    skipped_bm25 += 1
            except:
                filtered_corpus.append(bm25_corpus[i])
                filtered_doc_map.append(entry)
        
        if skipped_bm25 > 0:
            print(f"   ℹ️ BM25 跳过 {skipped_bm25} 条重复数据")
        
        # 准备全量语料
        full_corpus = []
        
        # 重新分词旧数据
        if existing_doc_map:
            print(f"   ...正在重新分词旧数据 ({len(existing_doc_map)} 条)...")
            for entry in existing_doc_map:
                full_corpus.append(list(jieba.cut_for_search(entry['text'])))
        
        full_corpus.extend(filtered_corpus)
        full_doc_map = existing_doc_map + filtered_doc_map
        
        bm25 = BM25Okapi(full_corpus)
        
        # 保存索引
        bm25_dir = os.path.dirname(BM25_INDEX_PATH)
        if bm25_dir and not os.path.exists(bm25_dir):
            os.makedirs(bm25_dir)
            print(f"   📁 创建目录: {bm25_dir}")
        
        print(f"💾 正在保存 BM25 索引到 {BM25_INDEX_PATH}...")
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump((bm25, full_doc_map), f)
        print(f"   --> BM25 索引保存完成（共 {len(full_doc_map)} 条文档）")
    
    # 收尾
    sql_conn.close()
    print("\n🎉 表格数据处理完成！")
    print(f"   📊 提取指标: {len(unique_metrics)} 条")
    print(f"   🧠 向量数据: {len(new_data) if milvus_data else 0} 条")
    print(f"   📑 BM25 索引: {len(full_doc_map) if bm25_corpus or existing_doc_map else 0} 条")


if __name__ == "__main__":
    main()
