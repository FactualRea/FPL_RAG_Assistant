from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = PROJECT_DIR / "vector_db"
COLLECTION_NAME = "fpl"
MODEL = "gpt-4.1-nano"
EMBEDDING_MODEL = "text-embedding-3-large"
load_dotenv(PROJECT_DIR / ".env")

SYSTEM_PROMPT = """
You are a helpful Fantasy Premier League data assistant.
Answer questions using only the retrieved FPL data. Be precise about player,
club, fixture, gameweek, prices, and statistics. The source is a timestamped
snapshot, so state that stats may have changed since it was indexed. If the
context does not support an answer, say you cannot confirm it from the snapshot.
Do not claim access to a user's private FPL team or live points unless those
are present in the retrieved data.

Retrieved FPL data:
{context}
"""


def _require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to this project's .env file.")
    return api_key


@lru_cache(maxsize=1)
def _get_retriever():
    if not VECTOR_DB_DIR.is_dir():
        raise RuntimeError("The FPL index is missing. Run `uv run python ingest.py` first.")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=_require_api_key())
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 7})


@lru_cache(maxsize=1)
def _get_llm():
    return ChatOpenAI(temperature=0, model=MODEL, api_key=_require_api_key())


def _resolve_query(query: str, history: list[dict[str, str]]) -> str:
    previous_user_message = next(
        (turn["content"] for turn in reversed(history) if turn.get("role") == "user"),
        None,
    )
    if previous_user_message:
        return f"{previous_user_message}\n{query}"
    return query


def answer_question(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, list[Document]]:
    if not query.strip():
        raise ValueError("Enter a question before submitting.")

    conversation = history or []
    documents = _get_retriever().invoke(_resolve_query(query, conversation))
    if not documents:
        return "I could not find matching FPL data in the current index.", []

    context = "\n\n".join(document.page_content for document in documents)
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    for turn in conversation:
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))

    response = _get_llm().invoke(messages)
    return str(response.content), documents
