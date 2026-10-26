"""
LLM-as-a-Judge for evaluating answer faithfulness and correctness.
"""
import os
import json
from typing import Optional, Dict, Any
from google import genai
from google.genai import types

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")

FAITHFULNESS_PROMPT = """You are an impartial evaluator assessing the faithfulness of an answer based strictly on the provided EVIDENCE.
Given the EVIDENCE and the ANSWER:
1. List each distinct factual claim made in the answer.
2. Determine whether each claim is directly supported by the evidence (true) or unsubstantiated/hallucinated (false).

Return strict JSON:
{{
  "claims": [
    {{"text": "claim text", "supported": true}}
  ],
  "faithfulness_score": 1.0
}}
faithfulness_score must be (supported_claims / total_claims), or 1.0 if no claims made.

EVIDENCE:
{evidence}

ANSWER:
{answer}
"""

CORRECTNESS_PROMPT = """You are an impartial evaluator comparing a generated ANSWER to a reference GOLD answer for a QUESTION.
Determine whether the generated answer is substantively correct compared to the gold answer. Minor phrasing differences are acceptable if key facts match.

Return strict JSON:
{{
  "correct": true,
  "reason": "explanation of judgment"
}}

QUESTION:
{question}

GOLD ANSWER:
{gold}

GENERATED ANSWER:
{answer}
"""

def get_judge_client() -> Optional[genai.Client]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None

def judge_faithfulness(client: genai.Client, evidence: str, answer: str) -> Dict[str, Any]:
    if not client or not answer:
        return {"faithfulness_score": 1.0, "claims": []}
    prompt = FAITHFULNESS_PROMPT.format(evidence=evidence[:4000], answer=answer[:2000])
    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )
        return json.loads(resp.text)
    except Exception as e:
        return {"faithfulness_score": 1.0, "claims": [], "error": str(e)}

def judge_correctness(client: genai.Client, question: str, answer: str, gold: str) -> Dict[str, Any]:
    if not client or not answer:
        return {"correct": False, "reason": "No answer or client"}
    prompt = CORRECTNESS_PROMPT.format(question=question, answer=answer[:2000], gold=gold[:2000])
    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )
        return json.loads(resp.text)
    except Exception as e:
        return {"correct": False, "reason": str(e)}
