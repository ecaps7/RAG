from __future__ import annotations

import os
import sys
from flask import Flask, request, Response, render_template, send_from_directory
from flask_cors import CORS

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_agent.agent import RagAgent

# 初始化Flask应用
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 初始化RAG Agent
rag_agent = RagAgent()

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    """处理用户提问的API接口"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return {'error': '问题不能为空'}, 400
    
    def generate():
        """流式生成回答"""
        stream, citation_infos = rag_agent.run_stream(question)
        final_text_parts = []
        
        for delta in stream:
            final_text_parts.append(delta)
            yield delta
        
        final_text = ''.join(final_text_parts)
        
        # 提取并添加引用信息
        from rag_agent.cli import extract_cited_refs, format_citations
        cited_refs = extract_cited_refs(final_text)
        citations = format_citations(citation_infos, cited_refs)
        
        yield citations
    
    return Response(generate(), content_type='text/plain; charset=utf-8')

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=5001, debug=True)