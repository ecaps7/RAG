import os
import time
import requests
from pathlib import Path
from typing import Optional
import dotenv

dotenv.load_dotenv()

API_URL = "https://www.datalab.to/api/v1/marker"
API_KEY = os.getenv("DATALAB_API_KEY")

def submit_and_poll_pdf_conversion(
  pdf_path: Path,
  output_format: Optional[str] = 'json',
  use_llm: Optional[bool] = True
):
  url = "https://www.datalab.to/api/v1/marker"

  #
  # Submit initial request
  #
  with open(pdf_path, 'rb') as f:
    form_data = {
        'file': (pdf_path.name, f, 'application/pdf'),
        "force_ocr": (None, True),
        "paginate": (None, True),
        'output_format': (None, output_format),
        "use_llm": (None, use_llm),
        "strip_existing_ocr": (None, False),
        "disable_image_extraction": (None, True)
    }
  
    headers = {"X-Api-Key": API_KEY}

    response = requests.post(url, files=form_data, headers=headers)
    data = response.json()
    
    # 检查 API 响应是否成功
    if "request_check_url" not in data:
      print(f"API Error: {data}")
      return None

  #
  # Poll for completion
  #
  max_polls = 1000
  check_url = data["request_check_url"]
  for i in range(max_polls):
    response = requests.get(check_url, headers=headers) # Need to include headers for API key
    check_result = response.json()

    if check_result['status'] == 'complete':
      #
      # Your processing is finished, you can do your post-processing!
      #
      converted_document = check_result[output_format]
      
      # Debug: 打印返回的所有字段
      print(f"API response keys: {check_result.keys()}")
      
      # 根据 PDF 文件名确定输出目录
      pdf_stem = pdf_path.stem  # 例如 "CMB-2025-q1"
      output_dir = Path(__file__).parent.parent.parent / "outputs" / pdf_stem
      output_dir.mkdir(parents=True, exist_ok=True)
      
      # 保存 JSON 格式
      import json
      output_file = output_dir / f"{pdf_stem}.json"
      output_file.write_text(json.dumps(converted_document, ensure_ascii=False, indent=2))
      print(f"Saved converted document to {output_file}")
      
      return converted_document
      
    elif check_result["status"] == "failed":
      print("Failed to convert, uh oh...")
      break
    else:
      print("Waiting 5 more seconds to re-check conversion status")
      time.sleep(5)


if __name__ == "__main__":
  # 默认转换 data/CMB-2025-h1.pdf
  pdf_path = Path(__file__).parent.parent.parent / "data" / "CMB-2025-h1.pdf"
  
  if not pdf_path.exists():
    print(f"Error: PDF file not found at {pdf_path}")
  else:
    print(f"Converting {pdf_path}...")
    submit_and_poll_pdf_conversion(pdf_path)