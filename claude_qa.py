import os
import logging
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512

SYSTEM_PROMPT = """You are a document assistant. Answer questions ONLY using the provided document excerpts below.

Rules:
- If a question has multiple parts, address each part separately.
- For each part: if the excerpts support an answer, state it and cite which excerpt supports it (e.g.
  "According to Section 2..." or "The document states..."). You may apply a policy from the excerpts to
  the specific situation asked about even if the wording doesn't match exactly, as long as you are not
  adding any fact the excerpts do not state.
- For any part the excerpts do not address, say exactly: "This information is not found in the uploaded
  documents." for that part only. Never guess, infer, or present an ungrounded answer as fact.
- If none of the excerpts contain information relevant to any part of the question, respond with exactly:
  "This information is not found in the uploaded documents." and nothing else.
- Be concise and accurate. Keep answers under 150 words unless the question requires more detail.
- Never make up information not present in the excerpts.
- Do not reference your training data or general knowledge."""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page_info = f"Page {chunk['page_num']}" if chunk.get("page_num") else "Document"
        parts.append(f"[Excerpt {i} — {page_info}]\n{chunk['chunk_text']}")
    return "\n\n".join(parts)


def answer_question(
    question: str,
    chunks: list[dict],
    doc_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not chunks:
        return {
            "answer": "No documents are available to answer your question. Please upload a document first.",
            "sources": [],
            "doc_used": doc_name or "none",
            "model": MODEL,
        }

    context = _build_context(chunks)
    doc_label = f" from '{doc_name}'" if doc_name else ""

    user_message = (
        f"Document excerpts{doc_label}:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Question: {question}"
    )

    if not api_key:
        # Demo fallback when no API key is configured
        return {
            "answer": (
                "⚠️ ANTHROPIC_API_KEY is not set. In a live demo, Claude would answer "
                f"your question using {len(chunks)} relevant excerpts found in the document. "
                "Please add your API key to the .env file to enable AI responses."
            ),
            "sources": chunks,
            "doc_used": doc_name or "unknown",
            "model": "fallback",
        }

    try:
        client = anthropic.Anthropic(api_key=api_key)
        create_kwargs = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        message = client.messages.create(**create_kwargs)
        answer = message.content[0].text.strip()
    except anthropic.AuthenticationError:
        answer = "Invalid API key. Please check your ANTHROPIC_API_KEY in .env."
    except anthropic.RateLimitError:
        answer = "Rate limit reached. Please wait a moment and try again."
    except Exception as exc:
        logger.error("Claude API error: %s", exc)
        answer = f"AI service error: {exc}"

    return {
        "answer": answer,
        "sources": [
            {
                "chunk_text": c["chunk_text"],
                "page_num": c.get("page_num", 1),
                "relevance_score": c.get("relevance_score", 0.0),
            }
            for c in chunks
        ],
        "doc_used": doc_name or "unknown",
        "model": MODEL,
    }
