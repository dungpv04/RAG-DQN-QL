import random

def embedding_function(query, doc):
    """Giả định hàm tính similarity"""
    return random.uniform(0, 1)

def rerank_function(doc):
    """Giả định hàm rerank"""
    return random.uniform(0, 1)

def get_retrieved_text(docs):
    """Giả định hàm lấy text từ docs"""
    return " ".join(docs) if docs else ""
