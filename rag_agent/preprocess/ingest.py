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

load_dotenv()

# ================= 配置区域 =================
# 数据库文件路径
SQL_DB_PATH = "database/financial_rag.db"
MILVUS_DB_PATH = "database/financial_vectors.db"
BM25_INDEX_PATH = "database/bm25_index.pkl"

# Embedding 配置
OLLAMA_MODEL = "qwen3-embedding:4b"
EMBEDDING_DIM = 2560  # Qwen3-Embedding 4B 的标准维度

# LLM 配置（用于结构化指标提取）- OpenAI-compatible API
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v3.2")

# 输入数据文件
FILE_TABLE = 'outputs/CITIC-2025-q1/CITIC-2025-q1-table.json'
FILE_TEXT = 'outputs/CITIC-2025-q1/CITIC-2025-q1-text.json'

# 需要提取结构化指标的表格类型（基于 section_path 关键词匹配）
# 设为 None 表示处理所有表格，让 LLM 自行判断
METRIC_TABLE_KEYWORDS = None  # 或设置为列表以启用过滤

# 明确排除的表格类型（不包含财务指标）
EXCLUDE_TABLE_KEYWORDS = [
    "股东情况", "股东信息", "董事", "监事", "高管",
    "公司治理", "关联交易", "重大事项", "公司基本情况"
]

# ================= 核心函数 =================

# LLM 提取财务指标的 Prompt 模板
METRIC_EXTRACTION_PROMPT = """你是一个专业的财务数据提取助手。请从以下 HTML 表格中提取关键财务指标。

## 公司信息
- 公司名称: {company_name}
- 股票代码: {stock_code}
- 报告期: {report_period}
- 表格来源: {table_id}

## HTML 表格内容
```html
{raw_code}
```

## 提取要求
1. 只提取**当期**（最新报告期）的数值，不要提取同比数据或上年数据
2. 每个指标包含: 指标名称、数值、单位
3. 数值请转换为纯数字（去除千分位逗号，负数用负号表示）
4. 常见单位：百万元、亿元、元、%、个百分点

## 重点提取的指标类型
- 收入类：营业收入、净利息收入、非利息净收入、手续费及佣金净收入
- 利润类：净利润、归属于股东的净利润、扣非净利润
- 每股指标：基本每股收益、稀释每股收益、每股净资产
- 资产类：总资产、贷款总额、存款总额
- 负债类：总负债
- 权益类：股东权益、归属于股东权益
- 盈利能力：净资产收益率(ROE)、总资产收益率(ROA)、净利差、净利息收益率
- 资产质量：不良贷款率、拨备覆盖率、关注贷款率
- 资本充足：核心一级资本充足率、一级资本充足率、资本充足率
- 现金流：经营活动现金流净额

## 输出格式
请严格按照以下 JSON 格式输出，不要输出任何其他内容：
```json
[
  {{"metric_name": "营业收入", "metric_value": 83751.00, "unit": "百万元"}},
  {{"metric_name": "净利润", "metric_value": 37286.00, "unit": "百万元"}}
]
```

如果表格中没有可提取的财务指标，返回空数组 `[]`。
"""


# 初始化 OpenAI-compatible 客户端
llm_client = None

def get_llm_client():
    """获取或初始化 LLM 客户端"""
    global llm_client
    if llm_client is None:
        if not LLM_API_KEY:
            raise ValueError("❌ 未设置 LLM_API_KEY 或 DEEPSEEK_API_KEY 环境变量")
        llm_client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_API_BASE
        )
    return llm_client


def extract_metrics_with_llm(table_item: dict) -> list[dict]:
    """
    使用 LLM 从表格中提取结构化财务指标
    
    Args:
        table_item: 表格数据项，包含 raw_code, document_context 等
        
    Returns:
        提取的指标列表，每项包含 metric_name, metric_value, unit
    """
    try:
        # 获取表格元数据
        doc_ctx = table_item.get('document_context', {})
        metadata = table_item.get('metadata', {})
        
        company_name = doc_ctx.get('company_short', doc_ctx.get('company_name', '未知'))
        stock_code = doc_ctx.get('stock_code', '未知')
        report_period = doc_ctx.get('report_period', '未知')
        table_id = table_item.get('id', 'unknown')
        raw_code = metadata.get('raw_code', '')
        
        if not raw_code:
            return []
        
        # 检查是否是需要提取指标的表格类型
        section_path = metadata.get('section_path', [])
        section_str = ' '.join(section_path)
        
        # 排除明确不包含财务指标的表格
        if any(kw in section_str for kw in EXCLUDE_TABLE_KEYWORDS):
            return []
        
        # 如果设置了关键词过滤，检查是否匹配
        if METRIC_TABLE_KEYWORDS is not None:
            should_extract = any(kw in section_str for kw in METRIC_TABLE_KEYWORDS)
            if not should_extract:
                # 也检查表格摘要内容
                content = table_item.get('content', '')
                should_extract = any(kw in content for kw in ['营业收入', '净利润', '每股收益', '资本充足率', '不良贷款率'])
            
            if not should_extract:
                return []
        
        # 构建 prompt
        prompt = METRIC_EXTRACTION_PROMPT.format(
            company_name=company_name,
            stock_code=stock_code,
            report_period=report_period,
            table_id=table_id,
            raw_code=raw_code
        )
        
        # 调用 OpenAI-compatible API
        client = get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # 低温度确保输出稳定
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 解析 JSON 输出
        # 尝试提取 JSON 数组（可能被 ```json 包裹）
        json_match = re.search(r'\[[\s\S]*?\]', result_text)
        if json_match:
            metrics = json.loads(json_match.group())
            
            # 添加公司和报告期信息
            for m in metrics:
                m['stock_code'] = stock_code.split('/')[0].strip() if '/' in stock_code else stock_code
                m['company_name'] = company_name
                m['report_period'] = _normalize_report_period(report_period)
                m['source_table_id'] = table_id
            
            return metrics
        
        return []
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失败 ({table_item.get('id', 'unknown')}): {e}")
        return []
    except Exception as e:
        print(f"❌ LLM 提取失败 ({table_item.get('id', 'unknown')}): {e}")
        return []


def _normalize_report_period(period: str) -> str:
    """
    标准化报告期格式
    例如: "2025年第一季度" -> "2025-Q1"
    """
    period = period.strip()
    
    # 匹配 "2025年第一季度" 格式
    q_match = re.search(r'(\d{4})年第([一二三四])季度', period)
    if q_match:
        year = q_match.group(1)
        q_map = {'一': 'Q1', '二': 'Q2', '三': 'Q3', '四': 'Q4'}
        quarter = q_map.get(q_match.group(2), 'Q1')
        return f"{year}-{quarter}"
    
    # 匹配 "2025年半年度" 格式
    h_match = re.search(r'(\d{4})年半年度|(\d{4})年上半年', period)
    if h_match:
        year = h_match.group(1) or h_match.group(2)
        return f"{year}-H1"
    
    # 匹配 "2025年年度" 格式
    y_match = re.search(r'(\d{4})年年度|(\d{4})年度', period)
    if y_match:
        year = y_match.group(1) or y_match.group(2)
        return f"{year}-FY"
    
    return period


def get_embedding(text):
    """
    调用本地 Ollama 生成 BGE-M3 向量
    """
    try:
        # 简单清洗，避免空字符报错
        text = text.replace("\n", " ").strip()
        if not text:
            return np.zeros(EMBEDDING_DIM).tolist()
            
        response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
        embedding = response.get('embedding')
        
        if not embedding or len(embedding) != EMBEDDING_DIM:
            print(f"⚠️ 警告: 向量维度异常或为空，返回零向量。Text: {text[:20]}...")
            return np.zeros(EMBEDDING_DIM).tolist()
            
        return embedding
    except Exception as e:
        print(f"❌ Embedding 调用失败: {e}")
        return np.zeros(EMBEDDING_DIM).tolist()

def init_sqlite():
    """初始化 SQLite 结构化指标库"""
    print("🛠️ 正在初始化 SQLite...")
    
    # 确保数据库目录存在
    db_dir = os.path.dirname(SQL_DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"   📁 创建目录: {db_dir}")
    
    conn = sqlite3.connect(SQL_DB_PATH)
    cursor = conn.cursor()
    # 创建指标表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT,
            company_name TEXT,
            report_period TEXT,
            metric_name TEXT,
            metric_value REAL,
            unit TEXT,
            source_table_id TEXT
        )
    ''')
    # 清空旧数据（开发测试用）
    # cursor.execute('DELETE FROM financial_metrics') # 注释掉以支持追加模式
    conn.commit()
    return conn

def init_milvus():
    """初始化 Milvus 向量库"""
    print("🛠️ 正在初始化 Milvus Lite...")
    
    # 确保数据库目录存在
    db_dir = os.path.dirname(MILVUS_DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"   📁 创建目录: {db_dir}")
    
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    collection_name = "financial_chunks"
    if client.has_collection(collection_name):
        print(f"   ℹ️ 集合 {collection_name} 已存在，将追加数据。")
    else:
        client.create_collection(
            collection_name=collection_name,
            dimension=EMBEDDING_DIM,
            metric_type="COSINE", # 余弦相似度
            auto_id=True          # 自动生成主键 ID
        )
    return client, collection_name

# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(
        description="ingest.py: 处理财报数据，生成结构化指标、向量和关键词索引"
    )

    parser.add_argument("--input-text-file", type=str, help="输入文本数据文件路径（JSON格式）")
    parser.add_argument("--input-table-file", type=str, help="输入表格数据文件路径（JSON格式）")
    args = parser.parse_args()

    global FILE_TEXT, FILE_TABLE
    if args.input_text_file:
        FILE_TEXT = args.input_text_file
    if args.input_table_file:
        FILE_TABLE = args.input_table_file

    # 1. 检查文件是否存在
    if not os.path.exists(FILE_TABLE) or not os.path.exists(FILE_TEXT):
        print("❌ 错误：未找到 JSON 数据文件，请确保文件名正确。")
        return

    # 2. 初始化数据库
    sql_conn = init_sqlite()
    sql_cursor = sql_conn.cursor()
    milvus_client, collection_name = init_milvus()

    # 3. 加载数据
    print("📂 正在读取 JSON 文件...")
    with open(FILE_TABLE, 'r', encoding='utf-8') as f:
        table_data = json.load(f)
    with open(FILE_TEXT, 'r', encoding='utf-8') as f:
        text_data = json.load(f)

    # ---------------------------------------------------------
    # A. SQL Layer: 使用 LLM 自动提取结构化指标
    # 从表格的 raw_code (HTML) 中解析关键财务数据
    # ---------------------------------------------------------
    print("📊 正在使用 LLM 提取结构化指标...")
    
    all_metrics = []
    tables_processed = 0
    
    for item in table_data:
        metrics = extract_metrics_with_llm(item)
        if metrics:
            all_metrics.extend(metrics)
            tables_processed += 1
            print(f"   ✓ {item.get('id', 'unknown')}: 提取了 {len(metrics)} 个指标")
    
    # 去重（同一指标可能在多个表格中出现）
    seen = set()
    unique_metrics = []
    for m in all_metrics:
        key = (m['stock_code'], m['report_period'], m['metric_name'])
        if key not in seen:
            seen.add(key)
            unique_metrics.append(m)
    
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
                m['source_table_id']
            )
            for m in unique_metrics
        ]
        
        sql_cursor.executemany('''
            INSERT INTO financial_metrics 
            (stock_code, company_name, report_period, metric_name, metric_value, unit, source_table_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', sql_records)
        sql_conn.commit()
        print(f"   --> 从 {tables_processed} 个表格中提取并存入 {len(unique_metrics)} 条指标（去重后）")

    # ---------------------------------------------------------
    # B. Vector + Keyword Layer: 处理表格和文本
    # ---------------------------------------------------------
    print("🧠 正在处理向量 Embedding 和 BM25 分词 (这可能需要几分钟)...")
    
    milvus_data = []
    bm25_corpus = []  # 存放分词后的列表
    doc_map = []      # 存放索引 ID 到 原始数据的映射

    # --- 处理表格 (Tables) ---
    for item in table_data:
        content = item.get('content', '') # 表格摘要
        if not content: continue

        # 1. 生成向量
        vec = get_embedding(content)
        
        # 2. 准备 Milvus 数据
        # 注意：Milvus Lite 的 metadata 只能存基本类型，这里把复杂结构转 JSON 字符串
        metadata_dict = {
            "source_id": str(item.get('id', 'unknown')),
            "type": "table",
            "page": str(item.get('page', 0)),
            # 关键：表格存 raw_code (HTML)，这是给 LLM 看的
            "raw_data": item['metadata'].get('raw_code', ''),
            "section": " > ".join(item.get('metadata', {}).get('section_path', []))
        }
        
        entry = {
            "vector": vec,
            "text": content, # 存摘要用于语义搜索
            "subject": "table_summary",
            "metadata": json.dumps(metadata_dict, ensure_ascii=False)
        }
        milvus_data.append(entry)
        
        # 3. 准备 BM25 数据
        tokens = list(jieba.cut_for_search(content))
        bm25_corpus.append(tokens)
        doc_map.append(entry) # 记录映射，BM25 搜索得到 index 后可以查回 entry

    # --- 处理文本 (Text Chunks) ---
    for item in text_data:
        content = item.get('content', '')
        if not content: continue

        # 1. 生成向量
        vec = get_embedding(content)
        
        # 2. 准备 Milvus 数据
        metadata_dict = {
            "source_id": str(item.get('id', 'unknown')),
            "type": "text",
            "page": str(item.get('page', 0)),
            # 关键：文本直接存 content
            "raw_data": content, 
            "section": " > ".join(item.get('section_path', []))
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
    
    # 1. 写入 Milvus
    if milvus_data:
        print(f"💾 正在写入 Milvus ({len(milvus_data)} 条数据)...")
        res = milvus_client.insert(collection_name=collection_name, data=milvus_data)
        print(f"   --> 成功写入 {res['insert_count']} 条向量。")
    
    # 2. 构建并保存 BM25
    # 尝试加载已有索引以支持追加
    existing_doc_map = []
    if os.path.exists(BM25_INDEX_PATH):
        try:
            with open(BM25_INDEX_PATH, 'rb') as f:
                _, existing_doc_map = pickle.load(f)
            print(f"   ℹ️ 发现已有 BM25 索引，包含 {len(existing_doc_map)} 条文档，将合并新数据。")
        except Exception as e:
            print(f"   ⚠️ 读取已有 BM25 索引失败: {e}")

    if bm25_corpus or existing_doc_map:
        print("📑 正在构建/更新 BM25 索引...")
        from rank_bm25 import BM25Okapi
        
        # 准备全量语料
        full_corpus = []
        
        # 1. 处理旧数据 (需要重新分词，因为没存分词结果)
        if existing_doc_map:
            print(f"   ...正在重新分词旧数据 ({len(existing_doc_map)} 条)...")
            for entry in existing_doc_map:
                full_corpus.append(list(jieba.cut_for_search(entry['text'])))
        
        # 2. 添加新数据 (已经分好词了)
        full_corpus.extend(bm25_corpus)
        
        # 3. 合并 doc_map
        full_doc_map = existing_doc_map + doc_map
        
        bm25 = BM25Okapi(full_corpus)
        
        # 确保目录存在
        bm25_dir = os.path.dirname(BM25_INDEX_PATH)
        if bm25_dir and not os.path.exists(bm25_dir):
            os.makedirs(bm25_dir)
            print(f"   📁 创建目录: {bm25_dir}")
        
        # 保存 BM25 对象和映射关系
        print(f"💾 正在保存 BM25 索引到 {BM25_INDEX_PATH}...")
        with open(BM25_INDEX_PATH, 'wb') as f:
            # 我们只存 BM25模型 和 必要的映射，减少文件体积
            # doc_map 包含原始文本，如果太大可以只存 id，这里为了演示方便存了全文
            pickle.dump((bm25, full_doc_map), f)
        print("   --> BM25 索引保存完成。")

    # 收尾
    sql_conn.close()
    print("\n🎉 所有数据入库完成！请运行 search.py 进行测试。")

if __name__ == "__main__":
    main()