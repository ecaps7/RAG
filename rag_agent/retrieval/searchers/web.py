"""
联网搜索器 - 基于 Tavily 的实时网络搜索
"""

from typing import List
import os

from langchain_community.tools.tavily_search import TavilySearchResults

from .base import BaseSearcher
from ..types import SearchResult
from ...utils.logging import get_logger


class WebSearcher(BaseSearcher):
    """
    Tavily 联网搜索器
    
    使用 Tavily API 进行实时网络搜索，获取最新信息
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.logger = get_logger("WebSearcher")
        
        # 初始化 Tavily 搜索工具
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            self.logger.warning("未设置 TAVILY_API_KEY，联网搜索功能将不可用")
            self.search_tool = None
        else:
            self.search_tool = TavilySearchResults(
                max_results=max_results,
                api_key=tavily_api_key
            )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        执行联网搜索

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        if not self.search_tool:
            self.logger.warning("Tavily 搜索工具未初始化，返回空结果")
            return []

        try:
            # 执行搜索
            results = self.search_tool.invoke({"query": query})
            
            # 转换为 SearchResult 格式
            search_results = []
            for idx, result in enumerate(results[:top_k]):
                search_results.append(
                    SearchResult(
                        id=f"web_{idx}",
                        content=result.get("content", ""),
                        score=1.0 - (idx * 0.1),  # 简单的递减评分
                        source="web",
                        metadata={
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "doctype": "web"
                        }
                    )
                )
            
            self.logger.info(f"联网搜索返回 {len(search_results)} 条结果")
            return search_results

        except Exception as e:
            self.logger.error(f"联网搜索失败: {e}")
            return []
