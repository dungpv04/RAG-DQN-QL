import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def embedding_function(query, doc):
    """Trả về độ tương đồng 0–1 giữa query và doc bằng TF-IDF + cosine similarity"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([query, doc])  # 2 hàng
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0,0]
    return float(sim)

def rerank_function(doc: str) -> float:
    """
    Tính điểm rerank dựa trên độ dài của doc.
    Trả về số trong khoảng 0–1 (chuẩn hoá).
    """
    max_len = 1000  # giả định độ dài tối đa để chuẩn hoá
    score = min(len(doc) / max_len, 1.0)
    return score


def get_retrieved_text(docs):
    """Giả định hàm lấy text từ docs"""
    return " ".join(docs) if docs else ""
