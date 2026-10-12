"""
Knowledge Base Indexer and Semantic Retrieval for Task 2 Support Assistant.
Chunks markdown documents by section headers and provides precise section-level retrieval.
"""

import os
import re
from typing import List, Dict, Any

class KnowledgeBase:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.sections: List[Dict[str, Any]] = []
        self._load_and_index()

    def _load_and_index(self):
        """Loads all markdown documents and splits into section-level indexed chunks."""
        self.sections = []
        for fname in sorted(os.listdir(self.docs_dir)):
            if fname.endswith(".md"):
                fpath = os.path.join(self.docs_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract main document title
                doc_title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                doc_title = doc_title_match.group(1).strip() if doc_title_match else fname

                # Split on level 2 headers: ## Section X.Y: ...
                raw_sections = re.split(r"(?=^##\s+)", content, flags=re.MULTILINE)
                for sec in raw_sections:
                    sec = sec.strip()
                    if not sec or sec.startswith("# "):
                        continue

                    header_match = re.search(r"^##\s+(Section\s+[\d\.]+:?\s*[^:\n]+)", sec, re.MULTILINE)
                    if header_match:
                        sec_header = header_match.group(1).strip()
                    else:
                        first_line = sec.split("\n")[0].replace("#", "").strip()
                        sec_header = first_line

                    # Body content without header
                    body_lines = sec.split("\n")[1:]
                    sec_body = "\n".join(body_lines).strip()

                    self.sections.append({
                        "filename": fname,
                        "doc_title": doc_title,
                        "section_header": sec_header,
                        "content": sec_body,
                        "full_text": f"{doc_title} | {sec_header}\n{sec_body}"
                    })

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Hybrid lexical and keyword overlap search over section chunks.
        """
        q_tokens = set(re.findall(r"\w+", query.lower()))
        # Remove common stop words
        stopwords = {"what", "is", "the", "for", "and", "or", "to", "in", "a", "of", "how", "does", "can", "if", "an", "on", "are", "do"}
        q_tokens = q_tokens - stopwords

        scored_sections = []
        for sec in self.sections:
            text_tokens = set(re.findall(r"\w+", sec["full_text"].lower()))
            overlap = len(q_tokens.intersection(text_tokens))
            
            # Boost matches in section header or filename
            header_tokens = set(re.findall(r"\w+", sec["section_header"].lower()))
            header_overlap = len(q_tokens.intersection(header_tokens))
            
            total_score = overlap + (header_overlap * 3)
            scored_sections.append((total_score, sec))

        scored_sections.sort(key=lambda x: x[0], reverse=True)
        return [sec for score, sec in scored_sections[:top_k] if score > 0]

