# FPL RAG Assistant

A Gradio chat assistant that retrieves current Fantasy Premier League player, club, gameweek, and fixture data. It fetches read-only snapshots from the public FPL API and indexes them locally with Chroma.

## Requirements

- Python 3.12
- `uv`
- An OpenAI API key for embeddings and chat responses
- Internet access to refresh FPL data

## Setup (PowerShell)

From this folder:

```powershell
uv venv --python 3.12
. .\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your OpenAI key to `.env`. Do not commit `.env`.

## Refresh the FPL index

```powershell
uv run python ingest.py
```

Ingestion reads the public bootstrap and fixture endpoints, then stores a timestamped snapshot in `vector_db/`. Re-run after gameweeks or whenever you want fresher stats.

## Start the chat

```powershell
uv run python app.py
```

Open the local URL printed by Gradio. The bot answers from the indexed snapshot and shows retrieved FPL sources beside the conversation.

The FPL API data is public and does not require FPL account credentials. `OPENAI_API_KEY` is only used for vector embeddings and chat generation.
