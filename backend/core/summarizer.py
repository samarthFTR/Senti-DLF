"""
summarizer.py
-------------
Generates a natural-language summary of batch sentiment analysis results
using a deterministic, highly-polished templating engine.

This approach guarantees professional, grammatically perfect, and accurate
textual summaries based directly on the extracted insights, without the
hallucination or repetition risks of small seq2seq generative models.

Public API
----------
  get_summarizer()       : Returns the singleton SentimentSummarizer instance.
  SentimentSummarizer    : Exposes .summarize() to generate the paragraph.
"""

from __future__ import annotations

import logging
from typing import List, Optional

log = logging.getLogger("uvicorn.error")

class SentimentSummarizer:
    _instance: Optional["SentimentSummarizer"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def summarize(
        self,
        total: int,
        pos_pct: float,
        neg_pct: float,
        insights: List[dict],
    ) -> str:
        """
        Generate a 2-3 sentence natural-language summary.

        Args:
            total     : Total number of reviews analysed.
            pos_pct   : Percentage of positive reviews (0-100).
            neg_pct   : Percentage of negative/critical reviews (0-100).
            insights  : List of AspectInsight dicts with keys:
                        aspect, sentiment, mention_count, message.

        Returns:
            A highly polished summary paragraph string.
        """
        if total == 0:
            return "No reviews analyzed."

        pos_str = f"{pos_pct:.1f}%"
        neg_str = f"{neg_pct:.1f}%"

        # Categorise insights by dominant sentiment
        positives = [i for i in insights if i["sentiment"] == "positive"]
        negatives = [i for i in insights if i["sentiment"] == "negative"]
        mixed     = [i for i in insights if i["sentiment"] == "mixed"]

        parts: List[str] = [
            f"Analysis of {total} reviews reveals that {pos_str} of customers expressed positive sentiment, "
            f"while {neg_str} provided critical or negative feedback."
        ]

        # Handle strengths
        if positives:
            topics = [f"'{i['aspect'].lower()}'" for i in positives[:3]]
            if len(topics) > 1:
                topics_str = ", ".join(topics[:-1]) + f", and {topics[-1]}"
            else:
                topics_str = topics[0]
            parts.append(f"Customers are generally satisfied with aspects like {topics_str}.")

        # Handle concerns
        if negatives:
            topics = [f"'{i['aspect'].lower()}'" for i in negatives[:3]]
            if len(topics) > 1:
                topics_str = ", ".join(topics[:-1]) + f", and {topics[-1]}"
            else:
                topics_str = topics[0]
            parts.append(f"However, there are notable concerns regarding {topics_str}.")
            
        # Handle mixed feelings if there were no strong positives/negatives, or just to add color
        if not positives and not negatives and mixed:
            topics = [f"'{i['aspect'].lower()}'" for i in mixed[:3]]
            if len(topics) > 1:
                topics_str = ", ".join(topics[:-1]) + f", and {topics[-1]}"
            else:
                topics_str = topics[0]
            parts.append(f"Feedback is largely mixed, particularly concerning {topics_str}.")

        return " ".join(parts)


def get_summarizer() -> SentimentSummarizer:
    return SentimentSummarizer()
