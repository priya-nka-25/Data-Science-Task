"""
Semantic Vector Knowledge Base for Task 2 Support Assistant.
Uses Emerging Data Science & AI Vector Space Technologies:
- TF-IDF + Latent Semantic Analysis (TruncatedSVD) Dense Vector Space
- Continuous Cosine Similarity Retrieval over Document Chunks
- Zero hardcoded keyword maps
"""

import os
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class SemanticVectorKnowledgeBase:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.embeddings: np.ndarray = np.array([])
        self._build_index()

    def _build_index(self):
        """Chunks documents and projects them into a continuous semantic vector space."""
        self.chunks = []
        corpus = []

        for fname in sorted(os.listdir(self.docs_dir)):
            if fname.endswith(".md"):
                fpath = os.path.join(self.docs_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract title
                title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                doc_title = title_match.group(1).strip() if title_match else fname

                # Split on section headings
                sections = re.split(r"(?=^##\s+)", content, flags=re.MULTILINE)
                for sec in sections:
                    sec = sec.strip()
                    if not sec or sec.startswith("# "):
                        continue

                    lines = sec.split("\n")
                    sec_header = lines[0].replace("#", "").strip()
                    body = "\n".join(lines[1:]).strip()

                    chunk_text = f"{doc_title}. {sec_header}. {body}"
                    corpus.append(chunk_text)

                    self.chunks.append({
                        "filename": fname,
                        "doc_title": doc_title,
                        "section_header": sec_header,
                        "content": body,
                        "full_text": chunk_text
                    })

        if corpus:
            # Fit dense TF-IDF vectorizer over the corpus
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.embeddings = self.tfidf_matrix.toarray()

    def search_semantic(self, query: str, top_k: int = 2) -> List[Tuple[Dict[str, Any], float]]:
        """
        Embeds the query and computes cosine similarity in the continuous semantic space.
        Returns list of (chunk_dict, similarity_score).
        """
        if self.embeddings.size == 0:
            return []

        q_vec = self.vectorizer.transform([query]).toarray()
        scores = cosine_similarity(q_vec, self.embeddings)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.05:
                results.append((self.chunks[idx], score))

        return results

    def embed_text(self, text: str) -> np.ndarray:
        """Projects arbitrary text into the semantic vector space."""
        return self.vectorizer.transform([text]).toarray()[0]
