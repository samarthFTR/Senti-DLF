"""
summarizer.py
-------------
Generates a natural-language summary of batch sentiment analysis results
using FLAN-T5-small (80M params) running on CPU.

The input is the structured aspect insights already extracted by the
predict route, serialised into a short prompt. FLAN-T5 decodes a
2-3 sentence paragraph without any fine-tuning required.

Public API
----------
  get_summarizer()       : Returns the singleton SentimentSummarizer instance.
  SentimentSummarizer    : Wraps the FLAN-T5 pipeline and exposes .summarize().
"""

from __future__ import annotations

import logging
from typing import List, Optional

log = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Singleton summarizer
# ---------------------------------------------------------------------------

class SentimentSummarizer:
    _instance: Optional["SentimentSummarizer"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        log.info("Loading FLAN-T5-small summarizer on CPU...")
        try:
            from transformers import pipeline
            # force CPU — GPU is fully occupied by BERT/DeBERTa
            self._pipe = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1,          # -1 = CPU
                max_new_tokens=120,
            )
            log.info("FLAN-T5-small loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load FLAN-T5-small: {e}")
            self._pipe = None

        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            Summary paragraph string, or a template fallback if the model
            failed to load.
        """
        if not insights:
            return self._template_fallback(total, pos_pct, neg_pct, [])

        if self._pipe is None:
            return self._template_fallback(total, pos_pct, neg_pct, insights)

        prompt = self._build_prompt(total, pos_pct, neg_pct, insights)
        try:
            output = self._pipe(prompt, do_sample=False)[0]["generated_text"]
            return output.strip()
        except Exception as e:
            log.warning(f"FLAN-T5 inference failed, using template fallback: {e}")
            return self._template_fallback(total, pos_pct, neg_pct, insights)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, total, pos_pct, neg_pct, insights) -> str:
        lines = [
            f"You analyzed {total} customer reviews.",
            f"{pos_pct:.0f}% were positive and {neg_pct:.0f}% were negative or critical.",
            "Key findings:",
        ]
        for ins in insights[:6]:
            lines.append(f"- {ins['aspect']}: {ins['sentiment']} ({ins['mention_count']} mentions). {ins['message']}")
        lines.append(
            "Write a concise 2-3 sentence summary of what customers think, "
            "highlighting the main strengths and concerns:"
        )
        return "\n".join(lines)

    def _template_fallback(self, total, pos_pct, neg_pct, insights) -> str:
        """Rule-based summary when the model is unavailable."""
        pos_str = f"{pos_pct:.0f}%"
        neg_str = f"{neg_pct:.0f}%"

        positives = [i for i in insights if i["sentiment"] == "positive"]
        negatives = [i for i in insights if i["sentiment"] == "negative"]

        parts: List[str] = [
            f"Across {total} reviews, {pos_str} expressed positive sentiment "
            f"while {neg_str} were negative or critical."
        ]

        if positives:
            topics = ", ".join(i["aspect"].lower() for i in positives[:2])
            parts.append(f"Customers were most satisfied with {topics}.")

        if negatives:
            topics = ", ".join(i["aspect"].lower() for i in negatives[:2])
            parts.append(f"Common concerns centred around {topics}.")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Singleton getter
# ---------------------------------------------------------------------------

def get_summarizer() -> SentimentSummarizer:
    return SentimentSummarizer()
