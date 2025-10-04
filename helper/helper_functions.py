# import random
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity


# def embedding_function(query, doc):
#     """Trả về độ tương đồng 0–1 giữa query và doc bằng TF-IDF + cosine similarity"""
#     vectorizer = TfidfVectorizer()
#     tfidf = vectorizer.fit_transform([query, doc])  # 2 hàng
#     sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0,0]
#     return float(sim)

# def rerank_function(doc: str) -> float:
#     """
#     Tính điểm rerank dựa trên độ dài của doc.
#     Trả về số trong khoảng 0–1 (chuẩn hoá).
#     """
#     max_len = 1000  # giả định độ dài tối đa để chuẩn hoá
#     score = min(len(doc) / max_len, 1.0)
#     return score


# def get_retrieved_text(docs):
#     """Giả định hàm lấy text từ docs"""
#     return " ".join(docs) if docs else ""


# ...existing code...
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Iterable, Optional, Union
import re

# ...existing code...
def _clean_text(text: str) -> str:
    """Basic cleanup to reduce noise for TF-IDF."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def embedding_function(query: str, doc: str) -> float:
    """
    Return cosine similarity (0-1) between query and doc using TF-IDF.
    - Robust to empty inputs.
    - Fits TF-IDF on the pair (query, doc) to compute similarity.
    """
    q = _clean_text(query)
    d = _clean_text(doc)
    if not q or not d:
        return 0.0
    try:
        vectorizer = TfidfVectorizer().fit([q, d])
        tfidf = vectorizer.transform([q, d])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
        # numerical safety
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        return 0.0

def batch_embedding_similarities(query: str, docs: Iterable[str]) -> List[float]:
    """
    Compute similarities between a single query and an iterable of docs.
    Returns a list of floats (same order as docs).
    This is more efficient than calling embedding_function repeatedly because
    it fits TF-IDF on the whole corpus (query + docs) at once.
    """
    docs_list = [d for d in docs if d is not None]
    q = _clean_text(query)
    if not q or not docs_list:
        return [0.0] * len(docs_list)
    corpus = [q] + [_clean_text(d) for d in docs_list]
    try:
        vectorizer = TfidfVectorizer().fit(corpus)
        tfidf = vectorizer.transform(corpus)
        q_vec = tfidf[0:1]
        doc_vecs = tfidf[1:]
        sims = cosine_similarity(q_vec, doc_vecs).flatten()
        sims = [float(np.clip(s, 0.0, 1.0)) for s in sims]
        return sims
    except Exception:
        return [0.0] * len(docs_list)

def rerank_function(doc: str) -> float:
    """
    Simple rerank score in [0,1] for a document.
    Combines:
      - normalized length (longer up to max_len gives higher score)
      - token density heuristic (more 'substantial' tokens -> higher)
    Backwards compatible with previous signature.
    """
    if not doc:
        return 0.0
    text = _clean_text(doc)
    tokens = [t for t in re.split(r"\W+", text) if t]
    if not tokens:
        return 0.0

    # length score (caps at max_len)
    max_len = 1000
    length_score = min(len(text) / max_len, 1.0)

    # token density: penalize lots of very short tokens (noise)
    avg_token_len = sum(len(t) for t in tokens) / len(tokens)
    density_score = np.tanh((avg_token_len - 3) / 5) * 0.5 + 0.5  # maps to ~0-1

    # final composite score
    score = 0.6 * length_score + 0.4 * density_score
    return float(np.clip(score, 0.0, 1.0))

def get_retrieved_text(docs: Optional[Iterable[Union[str, dict]]] = None, max_chars: int = 2000) -> str:
    """
    Safely join retrieved docs into a single string.
    - Accepts iterables of strings or dicts with a 'text' key.
    - Truncates to `max_chars` to avoid huge payloads.
    """
    if not docs:
        return ""
    parts = []
    for d in docs:
        if isinstance(d, dict):
            text = d.get("text") or d.get("content") or ""
        else:
            text = d or ""
        text = _clean_text(text)
        if text:
            parts.append(text)
    joined = " \n---\n ".join(parts)
    if len(joined) <= max_chars:
        return joined
    # preserve start and end context
    return joined[: max_chars - 200] + "\n...[truncated]...\n" + joined[-200:]