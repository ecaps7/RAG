"""
文本数据处理与入库脚本

功能：
1. 解析标准化文本数据（包含章节、实体、来源等元信息）
2. 使用 LLM 从文本中提取结构化财务指标
3. 生成向量 Embedding 并存入 Milvus
4. 构建 BM25 关键词索引
5. 将提取的指标存入 SQLite

使用示例：
    python process_text_ingest.py --input-file outputs/CMB-2025-q1/CMB-2025-q1-text.json
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

METRIC_EXTRACTION_PROMPT = """你是一个专业的财务数据提取助手。请从以下文本段落中提取关键财务指标和数据。

## 文档信息
- 来源: {entity}
- 报告期: {report_period}
- 章节: {section}

## 文本内容
{content}

## 提取要求
1. 只提取**当期**（最新报告期）的明确数值，不要提取同比/环比数据或上年数据
2. 每个指标包含: 指标名称、数值、单位
3. 数值请转换为纯数字（去除千分位逗号，负数用负号表示）
4. 常见单位：百万元、亿元、元、%、个百分点

## 重点提取的指标类型
- 收入类：营业收入、净利息收入、非利息净收入、手续费及佣金净收入
- 利润类：净利润、归属于股东的净利润
- 每股指标：基本每股收益、稀释每股收益、每股净资产
- 资产类：资产总额、贷款和垫款总额、客户存款总额
- 负债类：负债总额
- 权益类：股东权益
- 盈利能力：净资产收益率(ROE/ROAE)、总资产收益率(ROA/ROAA)、净利差
- 资产质量：不良贷款率、不良贷款余额、拨备覆盖率、贷款拨备率
- 资本充足：核心一级资本充足率、一级资本充足率、资本充足率

请将提取的指标以结构化格式返回。如果文本中没有可提取的财务指标，返回空列表。
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


def extract_metrics_from_text(text_item: Dict) -> List[Dict]:
    """
    使用 LLM 从文本段落中提取结构化财务指标（支持原生结构化输出）
    
    Args:
        text_item: 文本数据项，包含 content, entity, section 等
        
    Returns:
        提取的指标列表，每项包含 metric_name, metric_value, unit
    """
    try:
        content = text_item.get('content', '') or text_item.get('original_content', '')
        if not content or len(content.strip()) < 20:
            return []
        
        # 获取元信息
        entity = text_item.get('entity', '未知')
        section = text_item.get('section', '') or ' > '.join(text_item.get('section_path', []))
        source = text_item.get('source', '')
        
        # 尝试从来源或章节中提取报告期信息
        report_period = _extract_report_period(content, source, section)
        
        # 快速过滤：如果文本中没有数字，跳过
        if not re.search(r'\d+\.?\d*', content):
            return []
        
        # 构建 prompt
        prompt = METRIC_EXTRACTION_PROMPT.format(
            entity=entity,
            report_period=report_period,
            section=section,
            content=content
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
        stock_code = _extract_stock_code(entity, source)
        company_name = _normalize_company_name(entity, stock_code)
        for m in metrics:
            m['stock_code'] = stock_code
            m['company_name'] = company_name
            m['report_period'] = _normalize_report_period(report_period)
            m['source_text_id'] = text_item.get('id', 'unknown')
        
        return metrics
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失败 (文本块 {text_item.get('id', 'unknown')}): {e}")
        return []
    except Exception as e:
        print(f"❌ LLM 提取失败 (文本块 {text_item.get('id', 'unknown')}): {e}")
        return []


def _extract_report_period(content: str, source: str, section: str) -> str:
    """从内容、来源或章节中提取报告期"""
    # 优先从 source 中提取（如 "CMB-2025-q1"）
    if source:
        match = re.search(r'(\d{4})[-_](q[1-4]|Q[1-4]|h[1-2]|H[1-2]|year|annual)', source, re.IGNORECASE)
        if match:
            year = match.group(1)
            period = match.group(2).upper()
            if period.startswith('Q'):
                return f"{year}年第{['一','二','三','四'][int(period[1])-1]}季度"
            elif period.startswith('H'):
                return f"{year}年半年度"
            else:
                return f"{year}年年度"
    
    # 从章节标题中提取
    period_match = re.search(r'(二〇\d{2}|20\d{2})年(第[一二三四]季度|半年度|年度)', section)
    if period_match:
        return period_match.group(0)
    
    # 从内容中提取
    period_match = re.search(r'(20\d{2})年(第[一二三四]季度|半年度|年度|[1-3][-~]\d+月)', content)
    if period_match:
        return period_match.group(0)
    
    return "未知"


def _extract_stock_code(entity: str, source: str) -> str:
    """从实体名称或来源中提取股票代码"""
    # 从 source 中提取（如 "CMB" -> "600036.SH"）
    code_map = {
        'CMB': '600036.SH',
        '招商银行': '600036.SH',
        '招行': '600036.SH',
        'CITIC': '601998.SH',
        '中信银行': '601998.SH',
        '中信': '601998.SH'
    }
    
    for key, code in code_map.items():
        if key in (source or '') or key in (entity or ''):
            return code
    
    # 从 entity 中提取数字代码并添加 .SH 后缀
    code_match = re.search(r'\b(\d{6})\b', entity or '')
    if code_match:
        return f"{code_match.group(1)}.SH"
    
    # 如果实体名称中已包含完整代码（如 "601998.SH"）
    full_code_match = re.search(r'\b(\d{6}\.(SH|HK))\b', entity or '', re.IGNORECASE)
    if full_code_match:
        return full_code_match.group(1).upper()
    
    return entity or '未知'


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
        '招行': '招商银行',
        'CITIC': '中信银行',
        '中信银行股份有限公司': '中信银行',
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
        description="process_text_ingest.py: 处理标准化文本数据并入库"
    )
    parser.add_argument("--input-file", type=str, required=True,
                        help="输入文本数据文件路径（JSON格式）")
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
    
    # 3. 加载文本数据
    print(f"📂 正在读取文本文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    
    if not isinstance(text_data, list):
        print("❌ 错误：期望 JSON 格式为列表")
        return
    
    print(f"   ✓ 加载了 {len(text_data)} 个文本块")
    
    # ---------------------------------------------------------
    # A. SQL Layer: 使用 LLM 从文本中提取结构化指标
    # ---------------------------------------------------------
    print("📊 正在使用 LLM 提取结构化指标...")
    
    all_metrics = []
    texts_processed = 0
    
    for item in text_data:
        metrics = extract_metrics_from_text(item)
        if metrics:
            all_metrics.extend(metrics)
            texts_processed += 1
            print(f"   ✓ 文本块 {item.get('id', 'unknown')}: 提取了 {len(metrics)} 个指标")
    
    # 去重：同一公司、同一报告期、同一指标名称只保留一条
    # 如果有多个来源提取了相同指标，优先保留第一个
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
                m.get('source_text_id', 'unknown')
            )
            for m in unique_metrics
        ]
        
        sql_cursor.executemany('''
            INSERT OR REPLACE INTO financial_metrics 
            (stock_code, company_name, report_period, metric_name, metric_value, unit, source_table_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', sql_records)
        sql_conn.commit()
        print(f"   --> 从 {texts_processed} 个文本块中提取并存入 {len(unique_metrics)} 条指标（去重后）")
    else:
        print("   --> 未提取到任何指标")
    
    # ---------------------------------------------------------
    # B. Vector + Keyword Layer: 处理向量和 BM25
    # ---------------------------------------------------------
    print("🧠 正在处理向量 Embedding 和 BM25 分词...")
    
    milvus_data = []
    bm25_corpus = []
    doc_map = []
    
    for item in text_data:
        content = item.get('content', '') or item.get('original_content', '')
        if not content or len(content.strip()) < 10:
            continue
        
        # 1. 生成向量
        vec = get_embedding(content)
        
        # 2. 准备 Milvus 数据
        metadata_dict = {
            "source_id": str(item.get('id', 'unknown')),
            "type": "text",
            "page": str(item.get('page', 0)),
            "raw_data": content,
            "section": item.get('section', '') or ' > '.join(item.get('section_path', [])),
            "company_name": item.get('entity', ''),
            "stock_code": _extract_stock_code(item.get('entity', ''), item.get('source', '')),
            "report_period": _extract_report_period(content, item.get('source', ''), 
                                                     item.get('section', '')),
            "source": item.get('source', '')
        }
        
        entry = {
            "vector": vec,
            "text": content,
            "subject": "text_chunk",
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
    print("\n🎉 文本数据处理完成！")
    print(f"   📊 提取指标: {len(unique_metrics)} 条")
    print(f"   🧠 向量数据: {len(new_data) if milvus_data else 0} 条")
    print(f"   📑 BM25 索引: {len(full_doc_map) if bm25_corpus or existing_doc_map else 0} 条")


if __name__ == "__main__":
    main()
