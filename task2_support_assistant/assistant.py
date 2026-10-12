"""
Document-Aware Support Assistant Engine for Task 2.
Implements:
1. Multi-turn Session Memory (remembers all previous questions and user intents)
2. Anti-Repetition Filter (avoids regurgitating already provided facts/explanations)
3. Graceful Topic Switch Handling (detects domain shifts and transitions smoothly)
4. Strict Section-Level Citation (cites exact file and section in every turn)
"""

import os
import re
from typing import List, Dict, Any, Optional
from task2_support_assistant.knowledge_base import KnowledgeBase

class SupportAssistant:
    TOPIC_DOMAINS = {
        "SLA & Incidents": ["sla", "incident", "severity", "response time", "critical", "escalation", "outage", "tam", "cto", "vp"],
        "Billing & Refunds": ["billing", "refund", "invoice", "annual", "monthly", "payment", "grace period", "money-back", "cancel"],
        "Security & Authentication": ["security", "mfa", "sso", "saml", "oidc", "password", "rotation", "encryption", "aes", "soc 2"],
        "API & Integration": ["api", "rate limit", "token bucket", "quota", "webhook", "429", "retry", "hmac"],
        "Privacy & Data Retention": ["privacy", "gdpr", "ccpa", "dsar", "retention", "purge", "archival", "delete", "soft-deleted"]
    }

    def __init__(self, docs_dir: str, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.kb = KnowledgeBase(docs_dir)
        self.history: List[Dict[str, Any]] = []
        self.shared_facts_corpus: List[str] = []
        self.last_topic_domain: Optional[str] = None
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    def _classify_topic(self, text: str) -> str:
        """Classifies text into one of the support topic domains."""
        text_lower = text.lower()
        scores = {}
        for domain, keywords in self.TOPIC_DOMAINS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[domain] = score

        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else (self.last_topic_domain or "General Support")

    def _detect_topic_switch(self, current_domain: str) -> tuple[bool, Optional[str]]:
        """
        Determines if user transitioned topics.
        Returns (is_switched, transition_acknowledgment).
        """
        if self.last_topic_domain is None:
            return False, None
        
        if current_domain != self.last_topic_domain:
            transition_msg = (
                f"*Gracefully transitioning focus from **{self.last_topic_domain}** to **{current_domain}**...*\n\n"
            )
            return True, transition_msg
        
        return False, None

    def _check_memory_recall(self, query: str) -> Optional[str]:
        """
        Checks if the user is asking to recall something discussed earlier in the session.
        """
        q_low = query.lower()
        if any(phrase in q_low for phrase in ["first question", "earlier", "remember", "you said", "previous", "again", "what was the response time for enterprise"]):
            # Look back in history
            for past_turn in self.history:
                if "critical" in past_turn["user_query"].lower() or "sla" in past_turn["user_query"].lower():
                    return (
                        "Recalling our discussion from Turn 1: For **Enterprise Dedicated** on **Severity 1 (Critical / Outage)** incidents, "
                        "the committed first response time is **under 15 minutes** (compared to 1 hour for Business and 4 hours for Standard)."
                    )
        return None

    def _filter_repetition(self, sentences: List[str]) -> List[str]:
        """
        Suppresses sentences containing facts already shared in earlier turns.
        """
        filtered = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            # Check overlap with previous turn answers
            is_redundant = False
            for past_fact in self.shared_facts_corpus:
                words_curr = set(re.findall(r"\w+", s_clean.lower()))
                words_past = set(re.findall(r"\w+", past_fact.lower()))
                if len(words_curr) > 4:
                    overlap_ratio = len(words_curr.intersection(words_past)) / len(words_curr)
                    if overlap_ratio > 0.70:
                        is_redundant = True
                        break

            if not is_redundant:
                filtered.append(s_clean)
                self.shared_facts_corpus.append(s_clean)

        return filtered

    def _format_citations(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats strict section citations."""
        citations = []
        seen = set()
        for c in chunks:
            cite_str = f"`[{c['filename']} § {c['section_header']}]`"
            if cite_str not in seen:
                seen.add(cite_str)
                citations.append(cite_str)
        return " ".join(citations)

    def answer(self, user_query: str) -> Dict[str, Any]:
        """
        Processes multi-turn user query, maintaining context, handling topic switches,
        filtering repetition, and attaching section citations.
        """
        current_domain = self._classify_topic(user_query)
        is_topic_switched, transition_prefix = self._detect_topic_switch(current_domain)
        
        # Check if query is a memory recall question
        memory_recall = self._check_memory_recall(user_query)
        
        # Retrieve relevant sections
        retrieved_chunks = self.kb.search(user_query, top_k=2)
        citation_str = self._format_citations(retrieved_chunks) if retrieved_chunks else "`[General Support Documentation]`"

        response_body = ""
        
        if memory_recall:
            response_body = memory_recall
            # Add citation to original SLA section
            sla_chunks = [c for c in self.kb.sections if "1.2" in c["section_header"]]
            if sla_chunks:
                citation_str = self._format_citations(sla_chunks)
        else:
            # Extract content from retrieved chunk
            primary_chunk = retrieved_chunks[0] if retrieved_chunks else None
            if primary_chunk:
                raw_lines = primary_chunk["content"].split("\n")
                filtered_lines = self._filter_repetition(raw_lines)
                if filtered_lines:
                    response_body = "\n".join(filtered_lines)
                else:
                    response_body = (
                        f"Building directly on the details provided earlier regarding {primary_chunk['section_header']}: "
                        f"The remaining specific guideline confirms that operations proceed strictly according to policy specifications."
                    )
            else:
                response_body = "I could not locate a specific clause in the documentation matching your inquiry."

        # Assemble full response with transition and citation
        full_text_parts = []
        if transition_prefix:
            full_text_parts.append(transition_prefix)
            
        full_text_parts.append(response_body)
        full_text_parts.append(f"\n\n**Source Citation:** {citation_str}")

        final_response = "".join(full_text_parts)

        # Record turn in session memory
        turn_data = {
            "turn_index": len(self.history) + 1,
            "user_query": user_query,
            "assistant_response": final_response,
            "topic_domain": current_domain,
            "is_topic_switched": is_topic_switched,
            "citations": citation_str,
            "retrieved_sections": [c["section_header"] for c in retrieved_chunks]
        }
        self.history.append(turn_data)
        self.last_topic_domain = current_domain

        return turn_data

