// RAG Agent Web Frontend - Main Application Logic

// ============ 状态管理 ============

const state = {
    messages: [],
    currentAnswer: '',
    currentCitations: [],
    isLoading: false,
    error: null
};

// ============ DOM元素 ============

const messagesContainer = document.getElementById('messages-container');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');
const errorMessage = document.getElementById('error-message');
const statusIndicator = document.getElementById('status-indicator');

// ============ Markdown配置 ============

// 配置marked.js
marked.setOptions({
    gfm: true,
    breaks: true,
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return hljs.highlight(code, { language: lang }).value;
            } catch (err) {
                console.error('代码高亮失败:', err);
            }
        }
        return hljs.highlightAuto(code).value;
    }
});

// ============ 工具函数 ============

/**
 * 从文本中提取引用编号
 */
function extractCitedRefs(text) {
    const matches = text.matchAll(/\[(\d+)\]/g);
    return new Set([...matches].map(m => parseInt(m[1])));
}

/**
 * 生成唯一ID
 */
function generateId() {
    return Date.now().toString() + Math.random().toString(36).substr(2, 9);
}

/**
 * 转义HTML特殊字符
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 渲染Markdown并处理引用标记
 */
function renderMarkdown(text) {
    // 先渲染Markdown
    let html = marked.parse(text);
    
    // 将引用标记[n]转换为可点击链接
    html = html.replace(/\[(\d+)\]/g, (match, num) => {
        return `<a href="#citation-${num}" class="citation-mark" data-ref="${num}">[${num}]</a>`;
    });
    
    return html;
}

/**
 * 滚动到底部
 */
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * 更新状态指示器
 */
function updateStatus(status, isLoading = false) {
    const indicator = statusIndicator.querySelector('div');
    const text = statusIndicator.querySelector('span');
    
    if (isLoading) {
        indicator.className = 'w-2 h-2 bg-yellow-500 rounded-full mr-2 animate-pulse';
        text.textContent = status;
    } else {
        indicator.className = 'w-2 h-2 bg-green-500 rounded-full mr-2';
        text.textContent = status;
    }
}

/**
 * 显示错误消息
 */
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
    setTimeout(() => {
        errorMessage.classList.add('hidden');
    }, 5000);
}

// ============ 消息渲染 ============

/**
 * 创建用户消息元素
 */
function createUserMessage(content) {
    const wrapper = document.createElement('div');
    wrapper.className = 'flex justify-end mb-4';
    
    const bubble = document.createElement('div');
    bubble.className = 'user-message';
    bubble.textContent = content;
    
    wrapper.appendChild(bubble);
    return wrapper;
}

/**
 * 创建助手消息元素
 */
function createAssistantMessage(content, citations, isStreaming = false) {
    const wrapper = document.createElement('div');
    wrapper.className = 'flex justify-start mb-4';
    wrapper.id = isStreaming ? 'streaming-message' : `message-${generateId()}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'assistant-message';
    
    // 答案内容
    const answerDiv = document.createElement('div');
    answerDiv.className = 'markdown-content';
    answerDiv.innerHTML = renderMarkdown(content);
    
    // 添加光标(仅流式时)
    if (isStreaming) {
        const cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        answerDiv.appendChild(cursor);
    }
    
    bubble.appendChild(answerDiv);
    
    // 引用来源
    if (citations && citations.length > 0) {
        const citationsDiv = createCitationsElement(content, citations);
        bubble.appendChild(citationsDiv);
    }
    
    wrapper.appendChild(bubble);
    return wrapper;
}

/**
 * 创建引用来源元素
 */
function createCitationsElement(answerText, citations) {
    const container = document.createElement('div');
    container.className = 'mt-4 pt-4 border-t border-gray-200';
    
    // 标题
    const title = document.createElement('div');
    title.className = 'text-sm font-semibold text-gray-700 mb-3';
    title.innerHTML = '📊 数据来源 (References)';
    container.appendChild(title);
    
    // 提取实际引用的编号
    const citedRefs = extractCitedRefs(answerText);
    
    // 过滤引用列表
    const displayedCitations = citations.filter(c => citedRefs.has(c.ref));
    const filteredCount = citations.length - displayedCitations.length;
    
    // 如果没有检测到引用,显示全部
    const finalCitations = displayedCitations.length > 0 ? displayedCitations : citations;
    
    // 引用列表
    const listDiv = document.createElement('div');
    listDiv.className = 'space-y-2';
    
    finalCitations.forEach(citation => {
        const item = document.createElement('div');
        item.className = 'citation-item';
        item.id = `citation-${citation.ref}`;
        
        // 类型标签
        let badgeClass = 'badge-text';
        let badgeText = '文本';
        if (citation.doc_type === 'table') {
            badgeClass = 'badge-table';
            badgeText = '表格';
        } else if (citation.doc_type === 'sql') {
            badgeClass = 'badge-sql';
            badgeText = '结构化数据';
        }
        
        const badge = `<span class="badge ${badgeClass}">${badgeText}</span>`;
        
        // 页码信息
        const pageInfo = citation.page && citation.doc_type !== 'sql' 
            ? ` <span class="text-gray-500">(Page: ${citation.page})</span>` 
            : '';
        
        item.innerHTML = `
            <span class="font-medium text-blue-600">[${citation.ref}]</span>
            ${badge}
            <span class="text-gray-800">${escapeHtml(citation.title)}</span>
            ${pageInfo}
        `;
        
        // 点击引用条目时高亮对应的引用标记
        item.addEventListener('click', () => {
            highlightCitationMarks(citation.ref);
        });
        
        listDiv.appendChild(item);
    });
    
    container.appendChild(listDiv);
    
    // 过滤统计
    if (filteredCount > 0) {
        const filterInfo = document.createElement('div');
        filterInfo.className = 'text-sm text-gray-500 italic mt-3';
        filterInfo.textContent = `(已过滤 ${filteredCount} 条未引用的检索源)`;
        container.appendChild(filterInfo);
    }
    
    // 如果没有检测到引用标记
    if (displayedCitations.length === 0 && citations.length > 0) {
        const noRefInfo = document.createElement('div');
        noRefInfo.className = 'text-sm text-gray-500 italic mt-3';
        noRefInfo.textContent = '(未检测到引用标记,显示所有检索源)';
        container.appendChild(noRefInfo);
    }
    
    return container;
}

/**
 * 高亮引用标记
 */
function highlightCitationMarks(ref) {
    // 移除所有高亮
    document.querySelectorAll('.citation-item').forEach(el => {
        el.classList.remove('highlighted');
    });
    
    // 高亮当前条目
    const citationItem = document.getElementById(`citation-${ref}`);
    if (citationItem) {
        citationItem.classList.add('highlighted');
        
        // 滚动到引用条目
        citationItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

/**
 * 更新流式消息
 */
function updateStreamingMessage(content) {
    const streamingMsg = document.getElementById('streaming-message');
    if (streamingMsg) {
        const answerDiv = streamingMsg.querySelector('.markdown-content');
        answerDiv.innerHTML = renderMarkdown(content);
        
        // 添加光标
        const cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        answerDiv.appendChild(cursor);
    }
}

/**
 * 完成流式消息
 */
function finalizeStreamingMessage(citations) {
    const streamingMsg = document.getElementById('streaming-message');
    if (streamingMsg) {
        // 移除光标
        const cursor = streamingMsg.querySelector('.typing-cursor');
        if (cursor) cursor.remove();
        
        // 添加引用来源
        if (citations && citations.length > 0) {
            const bubble = streamingMsg.querySelector('.assistant-message');
            const answerDiv = streamingMsg.querySelector('.markdown-content');
            const citationsDiv = createCitationsElement(answerDiv.textContent, citations);
            bubble.appendChild(citationsDiv);
        }
        
        // 移除流式ID
        streamingMsg.id = `message-${generateId()}`;
        
        // 添加引用标记点击事件
        streamingMsg.querySelectorAll('.citation-mark').forEach(mark => {
            mark.addEventListener('click', (e) => {
                e.preventDefault();
                const ref = parseInt(mark.dataset.ref);
                highlightCitationMarks(ref);
            });
        });
    }
}

// ============ SSE客户端 ============

/**
 * 发送问题并处理流式响应
 */
async function sendQuestion(question) {
    // 更新状态
    state.isLoading = true;
    state.currentAnswer = '';
    state.currentCitations = [];
    state.error = null;
    
    // 禁用输入
    questionInput.disabled = true;
    sendButton.disabled = true;
    updateStatus('思考中...', true);
    
    // 添加用户消息
    const userMsg = createUserMessage(question);
    messagesContainer.appendChild(userMsg);
    
    // 添加流式助手消息占位符
    const assistantMsg = createAssistantMessage('', [], true);
    messagesContainer.appendChild(assistantMsg);
    scrollToBottom();
    
    try {
        // 发起SSE请求
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP错误: ${response.status}`);
        }
        
        // 读取SSE流
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // 处理SSE事件
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // 保留未完成的部分
            
            for (const line of lines) {
                if (!line.trim()) continue;
                
                const eventMatch = line.match(/^event: (.+)$/m);
                const dataMatch = line.match(/^data: (.+)$/m);
                
                if (eventMatch && dataMatch) {
                    const event = eventMatch[1];
                    const data = JSON.parse(dataMatch[1]);
                    
                    handleSSEEvent(event, data);
                }
            }
        }
        
    } catch (error) {
        console.error('请求失败:', error);
        showError(`请求失败: ${error.message}`);
        
        // 移除流式消息
        const streamingMsg = document.getElementById('streaming-message');
        if (streamingMsg) streamingMsg.remove();
        
    } finally {
        // 恢复输入
        state.isLoading = false;
        questionInput.disabled = false;
        sendButton.disabled = false;
        updateStatus('就绪', false);
    }
}

/**
 * 处理SSE事件
 */
function handleSSEEvent(event, data) {
    switch (event) {
        case 'delta':
            // 追加文本
            state.currentAnswer += data.content;
            updateStreamingMessage(state.currentAnswer);
            scrollToBottom();
            break;
            
        case 'citation':
            // 保存引用信息
            state.currentCitations = data.citations;
            break;
            
        case 'done':
            // 完成
            finalizeStreamingMessage(state.currentCitations);
            scrollToBottom();
            break;
            
        case 'error':
            // 错误
            showError(data.message);
            const streamingMsg = document.getElementById('streaming-message');
            if (streamingMsg) streamingMsg.remove();
            break;
    }
}

// ============ 事件处理 ============

/**
 * 发送按钮点击
 */
sendButton.addEventListener('click', () => {
    const question = questionInput.value.trim();
    if (!question) {
        showError('请输入问题');
        return;
    }
    
    if (question.length > 2000) {
        showError('问题长度不能超过2000字符');
        return;
    }
    
    // 清空输入框
    questionInput.value = '';
    
    // 发送问题
    sendQuestion(question);
});

/**
 * 回车键发送(Shift+Enter换行)
 */
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendButton.click();
    }
});

/**
 * 自动调整输入框高度
 */
questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = questionInput.scrollHeight + 'px';
});

// ============ 初始化 ============

console.log('RAG Agent Web Frontend 已加载');

// 检查服务健康状态
fetch('/api/health')
    .then(res => res.json())
    .then(data => {
        console.log('服务状态:', data);
        updateStatus('就绪', false);
    })
    .catch(err => {
        console.error('健康检查失败:', err);
        updateStatus('服务异常', false);
        showError('无法连接到后端服务');
    });
