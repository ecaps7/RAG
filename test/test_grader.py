#!/usr/bin/env python3
"""
Test script for document grader functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.agent import RagAgent


def test_grader():
    """Test document grader functionality."""
    print("Testing document grader...")
    
    # Initialize RAG agent
    agent = RagAgent()
    
    # Test question
    question = "2025年第一季度招商银行的净利润是多少？"
    
    # Run RAG pipeline
    answer = agent.run(question)
    
    print(f"\nQ    {question}")
    # Run RAG pipeline
    answer = agent.run(question)
    
    print(f"\nA    {answer.text}")
    print(f"\nConfidence: {answer.confidence}")
    print(f"\nMetadata: {answer.meta}")
    print(f"\nCitations: {answer.citations}")
    
    return True


if __name__ == "__main__":
    test_grader()