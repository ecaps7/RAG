import os, re, glob, json, hashlib, argparse
from typing import List, Dict, Iterable, Tuple, Optional
import fitz                     # PyMuPDF
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============== 可调参数 ===============
DEFAULT_INPUT_DIR = "data"
DEFAULT_OUTPUT_DIR = "outputs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TABLE_AS = "markdown"   # "markdown" 或 "json"
INCLUDE_EMPTY_TABLES = False
# =======================================

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def clean_text_page(raw: str) -> str:
    """
    轻量清洗：去空行、合并多空格；可按需增加页眉/页脚过滤规则。
    """
    if not raw:
        return ""
    # 去掉页码/页眉行
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]                 # 去空行
    # 过滤纯数字页码（如：1, 12, 123）
    lines = [ln for ln in lines if not re.fullmatch(r"\d+", ln)]
    # 过滤带"第X页"格式的页码
    lines = [ln for ln in lines if not re.fullmatch(r"第?\s*\d+\s*页?", ln)]
    txt = "\n".join(lines)
    return txt

def sha1(obj) -> str:
    m = hashlib.sha1()
    m.update(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return m.hexdigest()

def iter_pdf_files(root: str) -> Iterable[str]:
    for p in sorted(glob.glob(os.path.join(root, "**/*.pdf"), recursive=True)):
        yield p

def _detect_time_point(text: str) -> str:
    # 简易时点检测：年份/季度/月份/上半年/下半年
    patterns = [
        r"(20\d{2}年(?:[一二三四]季度|[上下]半年|\d{1,2}月)?)",
        r"(20\d{2}\s*Q[1-4])",
        r"(截至\s*20\d{2}年\d{1,2}月)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""


def _detect_unit(text: str) -> str:
    # 简易单位检测
    m = re.search(r"单位[：:】]?\s*([\u4e00-\u9fa5A-Za-z%bp亿万元]*)", text)
    if m:
        return m.group(1).strip()
    # 回退：根据常见词判断
    for u in ["亿元", "万元", "%", "bp", "千元", "亿元人民币"]:
        if u in text:
            return u
    return ""


def extract_text_docs(pdf_path: str) -> List[Dict]:
    """
    版面感知文本抽取：使用 PyMuPDF blocks，粗略识别双栏并保持阅读顺序。
    """
    docs = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            blocks = page.get_text("blocks") or []
            # blocks: (x0, y0, x1, y1, text, block_no, block_type)
            if not blocks:
                text = clean_text_page(page.get_text("text") or "")
                if not text:
                    continue
                time_point = _detect_time_point(text)
                unit = _detect_unit(text)
                docs.append({
                    "page_content": text,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "source_path": os.path.abspath(pdf_path),
                        "page": i + 1,
                        "doctype": "text",
                        "layout": "single_column",
                        "time_point": time_point,
                        "unit": unit,
                    }
                })
                continue

            # 简单两栏检测：按 x0 聚类为两组
            xs = sorted([b[0] for b in blocks])
            two_col = False
            if len(xs) > 4:
                # 若中位数与第 1/3 位差距较大，认为两栏
                mid = xs[len(xs)//2]
                left = xs[len(xs)//3]
                two_col = abs(mid - left) > 60  # 阈值可调

            if two_col:
                left_blocks = [b for b in blocks if b[0] <= min(xs) + (max(xs) - min(xs)) / 2]
                right_blocks = [b for b in blocks if b[0] > min(xs) + (max(xs) - min(xs)) / 2]
                left_blocks.sort(key=lambda b: (b[1], b[0]))
                right_blocks.sort(key=lambda b: (b[1], b[0]))
                text = "\n".join([clean_text_page(b[4]) for b in left_blocks + right_blocks])
                layout = "two_column"
            else:
                blocks.sort(key=lambda b: (b[1], b[0]))
                text = "\n".join([clean_text_page(b[4]) for b in blocks])
                layout = "single_column"

            text = clean_text_page(text)
            if not text:
                continue
            time_point = _detect_time_point(text)
            unit = _detect_unit(text)
            docs.append({
                "page_content": text,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "source_path": os.path.abspath(pdf_path),
                    "page": i + 1,
                    "doctype": "text",
                    "layout": layout,
                    "time_point": time_point,
                    "unit": unit,
                }
            })
    return docs

def table_to_markdown(rows: List[List[str]]) -> str:
    if not rows or not rows[0]:
        return ""
    # 规范化单元格
    norm_rows = [[norm_ws(c) if c is not None else "" for c in row] for row in rows]
    
    # 过滤掉完全空的行
    norm_rows = [r for r in norm_rows if any(cell.strip() for cell in r)]
    if not norm_rows:
        return ""
    
    # 检查第一行是否像表头（包含文字而非纯数字）
    first_row = norm_rows[0]
    is_header = any(cell and not cell.replace('.', '').replace(',', '').replace('-', '').replace('%', '').isdigit() 
                   for cell in first_row)
    
    if is_header:
        header = first_row
        body = norm_rows[1:]
    else:
        # 如果没有明显表头，使用列号作为表头
        header = [f"列{i+1}" for i in range(len(first_row))]
        body = norm_rows
    
    md = []
    md.append(" | ".join(header))
    md.append(" | ".join(["---"] * len(header)))
    for r in body:
        # 确保行的列数与表头一致
        padded_row = r + [""] * (len(header) - len(r))
        md.append(" | ".join(padded_row[:len(header)]))
    return "\n".join(md)

def extract_table_docs(pdf_path: str) -> List[Dict]:
    """
    用 pdfplumber 抽表格，并尽量识别表题、列名、单位与时点。
    - 优先使用 page.find_tables 获取 bbox，再结合 extract_words 搜索上方最近的标题行
    - 列名：采用首行作为表头（若不是，回退自动生成）
    """
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            }
            try:
                tbls = page.find_tables(settings) or []
            except Exception:
                tbls = []

            words = []
            try:
                words = page.extract_words(x_tolerance=1, y_tolerance=1) or []
            except Exception:
                words = []

            def nearest_title(bbox):
                x0, y0, x1, y1 = bbox
                cand = [w for w in words if w.get("bottom", 0) <= y0 and abs(y0 - w.get("bottom", 0)) < 60]
                # 聚合成行
                lines: Dict[int, List[str]] = {}
                for w in cand:
                    top = int(w.get("top", 0))
                    lines.setdefault(top, []).append(w.get("text", ""))
                if not lines:
                    return ""
                # 选最近一行
                best_top = max(lines.keys())
                line_text = " ".join(lines[best_top]).strip()
                # 优先包含“表”“Table”等关键词
                if re.search(r"(表\s*\d*[:：]?|table)\s*", line_text, re.IGNORECASE):
                    return line_text
                return line_text

            # 回退：extract_tables
            extracted_rows = []
            if not tbls:
                try:
                    extracted_rows = page.extract_tables(settings) or []
                except Exception:
                    extracted_rows = []

            # 统一为 [(bbox, rows)] 结构
            pairs: List[Tuple[Optional[Tuple[float, float, float, float]], List[List[str]]]] = []
            if tbls:
                for t in tbls:
                    try:
                        rows = t.extract()
                    except Exception:
                        rows = []
                    pairs.append((t.bbox, rows))
            else:
                for rows in extracted_rows:
                    pairs.append((None, rows))

            for t_idx, (bbox, rows) in enumerate(pairs):
                if (not rows) and (not INCLUDE_EMPTY_TABLES):
                    continue

                # 列名识别（基于首行）
                header = rows[0] if (rows and rows[0]) else []
                # 单位/时点：取标题行或页面文本中最近的行
                title_text = nearest_title(bbox) if bbox else ""
                unit = _detect_unit(title_text)
                time_point = _detect_time_point(title_text)

                if TABLE_AS == "json":
                    content = json.dumps({"rows": rows}, ensure_ascii=False)
                    repr_ = "json"
                else:
                    md = table_to_markdown(rows or [])
                    if not md and not INCLUDE_EMPTY_TABLES:
                        continue
                    content = f"[TABLE]\n{md}"
                    repr_ = "markdown"

                meta = {
                    "source": os.path.basename(pdf_path),
                    "source_path": os.path.abspath(pdf_path),
                    "page": i + 1,
                    "table_index": t_idx,
                    "doctype": "table",
                    "repr": repr_,
                    "row_count": len(rows) if rows else 0,
                }
                if header:
                    meta["columns"] = [norm_ws(h) for h in header]
                if title_text:
                    meta["table_name"] = norm_ws(title_text)
                if unit:
                    meta["unit"] = unit
                if time_point:
                    meta["time_point"] = time_point

                docs.append({
                    "page_content": content,
                    "metadata": meta,
                })
    return docs

def chunk_text_docs(docs: List[Dict],
                    chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    对正文文档进行切分；表格通常不切，保持为原子块。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", "；", ";", "：", ":"]
    )
    out = []
    for d in docs:
        if d["metadata"].get("doctype") == "table":
            out.append(d)
            continue
        for ch in splitter.split_text(d["page_content"]):
            item = {
                "page_content": ch,
                "metadata": dict(d["metadata"])  # 保留 layout/time_point/unit
            }
            out.append(item)
    return out

def dedup_docs(docs: List[Dict]) -> List[Dict]:
    """
    简单去重：基于文本+来源元信息的 sha1。
    """
    seen = set()
    uniq = []
    for d in docs:
        key = sha1({
            "t": d["page_content"],
            "s": d["metadata"].get("source_path"),
            "p": d["metadata"].get("page"),
            "dt": d["metadata"].get("doctype"),
            "ti": d["metadata"].get("table_index", None)
        })
        if key in seen:
            continue
        d["metadata"]["content_sha1"] = key
        uniq.append(d)
        seen.add(key)
    return uniq

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def dump_jsonl(path: str, rows: Iterable[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def preprocess(input_dir: str, output_dir: str,
               chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    ensure_dir(output_dir)

    all_text_docs: List[Dict] = []
    all_table_docs: List[Dict] = []

    pdfs = list(iter_pdf_files(input_dir))
    print(f"[INFO] Found {len(pdfs)} PDFs under {input_dir}")
    for pdf in pdfs:
        print(f"[INFO] Processing: {pdf}")
        text_docs = extract_text_docs(pdf)
        table_docs = extract_table_docs(pdf)
        all_text_docs.extend(text_docs)
        all_table_docs.extend(table_docs)

    # 切分（只切正文）
    print(f"[INFO] Chunking text docs ...")
    chunked_text_docs = chunk_text_docs(all_text_docs, chunk_size, chunk_overlap)

    # 表格保持原子块
    table_docs_final = all_table_docs

    # 去重
    print(f"[INFO] De-duplicating ...")
    chunked_text_docs = dedup_docs(chunked_text_docs)
    table_docs_final = dedup_docs(table_docs_final)

    # 产出
    text_out = os.path.join(output_dir, "text_chunks.jsonl")
    table_out = os.path.join(output_dir, "table_chunks.jsonl")
    all_out = os.path.join(output_dir, "all_chunks.jsonl")

    dump_jsonl(text_out, chunked_text_docs)
    dump_jsonl(table_out, table_docs_final)
    dump_jsonl(all_out, [*chunked_text_docs, *table_docs_final])

    print(f"[DONE] text_chunks:  {text_out}  ({len(chunked_text_docs)} records)")
    print(f"[DONE] table_chunks: {table_out}  ({len(table_docs_final)} records)")
    print(f"[DONE] all_chunks:   {all_out}  ({len(chunked_text_docs) + len(table_docs_final)} records)")

def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess PDFs (extract/clean/chunk) before embedding")
    ap.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, help="Folder containing PDFs")
    ap.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Folder to write JSONL outputs")
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--table_as", type=str, choices=["markdown", "json"], default=TABLE_AS,
                    help="How to serialize tables")
    return ap.parse_args()

def preprocess_pdfs(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    table_format: str = "markdown"
) -> Dict[str, str]:
    """
    可直接调用的预处理接口函数。
    
    Args:
        input_dir: PDF 文件所在目录
        output_dir: 输出目录
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        table_format: 表格格式 ("markdown" 或 "json")
    
    Returns:
        包含输出文件路径的字典
    """
    global TABLE_AS
    TABLE_AS = table_format
    preprocess(input_dir, output_dir, chunk_size, chunk_overlap)
    
    return {
        "text_chunks": os.path.join(output_dir, "text_chunks.jsonl"),
        "table_chunks": os.path.join(output_dir, "table_chunks.jsonl"),
        "all_chunks": os.path.join(output_dir, "all_chunks.jsonl")
    }


def main():
    global TABLE_AS
    args = parse_args()
    TABLE_AS = args.table_as
    preprocess(args.input_dir, args.output_dir, args.chunk_size, args.chunk_overlap)

if __name__ == "__main__":
    main()