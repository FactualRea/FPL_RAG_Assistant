from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = PROJECT_DIR / "vector_db"
COLLECTION_NAME = "fpl"
API_BASE = "https://fantasy.premierleague.com/api"
load_dotenv(PROJECT_DIR / ".env")

PLAYER_FIELDS = (
    "now_cost",
    "total_points",
    "points_per_game",
    "form",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    "ict_index",
)


def fetch_fpl_data() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fetch the public FPL bootstrap snapshot and all season fixtures."""
    session = requests.Session()
    session.headers.update({"User-Agent": "FPL-RAG-Assistant/1.0"})
    bootstrap_response = session.get(f"{API_BASE}/bootstrap-static/", timeout=30)
    bootstrap_response.raise_for_status()
    fixtures_response = session.get(f"{API_BASE}/fixtures/", timeout=30)
    fixtures_response.raise_for_status()
    return bootstrap_response.json(), fixtures_response.json()


def _display(value: Any, fallback: str = "unknown") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def build_documents(
    bootstrap: dict[str, Any],
    fixtures: list[dict[str, Any]],
    snapshot: str | None = None,
) -> list[Document]:
    """Convert FPL API records into source-tagged retrieval documents."""
    snapshot = snapshot or datetime.now(UTC).isoformat(timespec="seconds")
    teams = {team["id"]: team for team in bootstrap.get("teams", [])}
    positions = {position["id"]: position["singular_name"] for position in bootstrap.get("element_types", [])}
    events = {event["id"]: event for event in bootstrap.get("events", [])}
    current_event = next((event for event in events.values() if event.get("is_current")), None)
    upcoming_by_team: dict[int, list[dict[str, Any]]] = {team_id: [] for team_id in teams}

    for fixture in fixtures:
        if fixture.get("finished"):
            continue
        team_home_id = fixture.get("team_h")
        team_away_id = fixture.get("team_a")
        for team_id, opponent_id, home in (
            (team_home_id, team_away_id, True),
            (team_away_id, team_home_id, False),
        ):
            if team_id in upcoming_by_team and opponent_id in teams:
                upcoming_by_team[team_id].append(
                    {
                        "event": fixture.get("event"),
                        "kickoff_time": fixture.get("kickoff_time"),
                        "opponent": teams[opponent_id]["name"],
                        "venue": "home" if home else "away",
                    }
                )

    for schedule in upcoming_by_team.values():
        schedule.sort(key=lambda match: (match["event"] or 99, match["kickoff_time"] or ""))

    documents: list[Document] = []

    def add_document(identifier: str, source: str, text: str, kind: str) -> None:
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "id": identifier,
                    "source": source,
                    "kind": kind,
                    "snapshot": snapshot,
                },
            )
        )

    season = bootstrap.get("game_settings", {}).get("season", "current season")
    current_event_name = current_event.get("name", "not announced") if current_event else "not announced"
    current_event_id = current_event.get("id", "unknown") if current_event else "unknown"
    add_document(
        "fpl-overview",
        f"{API_BASE}/bootstrap-static/",
        (
            f"Fantasy Premier League season {season}. Snapshot collected at {snapshot}. "
            f"There are {len(bootstrap.get('elements', []))} players and "
            f"{len(teams)} Premier League clubs. Current gameweek: {current_event_name} "
            f"(gameweek {current_event_id}). Use this snapshot date when describing how current stats are."
        ),
        "overview",
    )

    for player in bootstrap.get("elements", []):
        team = teams.get(player.get("team"), {})
        position = positions.get(player.get("element_type"), "unknown position")
        stats = "; ".join(
            f"{field.replace('_', ' ')}: {_display(player.get(field))}"
            for field in PLAYER_FIELDS
        )
        schedule = upcoming_by_team.get(player.get("team"), [])[:3]
        next_matches = "; ".join(
            f"gameweek {match['event'] or 'TBC'} vs {match['opponent']} ({match['venue']}, {match['kickoff_time'] or 'kickoff TBC'})"
            for match in schedule
        ) or "no upcoming fixtures listed"
        text = (
            f"FPL player: {player.get('first_name', '')} {player.get('second_name', '')} "
            f"(known as {player.get('web_name', 'unknown')}). Position: {position}. "
            f"Club: {_display(team.get('name'))} ({_display(team.get('short_name'))}). "
            f"Status: {_display(player.get('status'))}. Availability news: "
            f"{_display(player.get('news'), 'no current news')}. Cost: {player.get('now_cost', 0) / 10:.1f}m. "
            f"Season statistics: {stats}. Upcoming fixtures: {next_matches}. "
            f"FPL data snapshot: {snapshot}."
        )
        add_document(
            f"fpl-player-{player['id']}",
            f"{API_BASE}/bootstrap-static/ player {player['id']}",
            text,
            "player",
        )

    for team_id, team in teams.items():
        schedule = upcoming_by_team.get(team_id, [])[:5]
        next_matches = "; ".join(
            f"gameweek {match['event'] or 'TBC'} vs {match['opponent']} ({match['venue']}, {match['kickoff_time'] or 'kickoff TBC'})"
            for match in schedule
        ) or "no upcoming fixtures listed"
        text = (
            f"Premier League club: {team['name']} ({team.get('short_name', '')}). "
            f"Overall strength: {team.get('strength', 'unknown')}; "
            f"home strength: {team.get('strength_overall_home', 'unknown')}; "
            f"away strength: {team.get('strength_overall_away', 'unknown')}. "
            f"Upcoming fixtures: {next_matches}. FPL data snapshot: {snapshot}."
        )
        add_document(
            f"fpl-team-{team_id}",
            f"{API_BASE}/bootstrap-static/ team {team_id}",
            text,
            "team",
        )

    for event_id, event in events.items():
        top_player = event.get("top_element_info") or {}
        text = (
            f"FPL gameweek {event_id}: {event.get('name', 'unnamed')}. "
            f"Deadline: {_display(event.get('deadline_time'))}. "
            f"Current: {event.get('is_current', False)}; next: {event.get('is_next', False)}; "
            f"finished: {event.get('finished', False)}. Average points: "
            f"{_display(event.get('average_entry_score'))}. Highest score: "
            f"{_display(event.get('highest_score'))}. Top player ID: "
            f"{_display(top_player.get('id'))}; points: {_display(top_player.get('points'))}. "
            f"FPL data snapshot: {snapshot}."
        )
        add_document(
            f"fpl-gameweek-{event_id}",
            f"{API_BASE}/bootstrap-static/ gameweek {event_id}",
            text,
            "gameweek",
        )

    for fixture in fixtures:
        home = teams.get(fixture.get("team_h"), {}).get("name", "unknown home team")
        away = teams.get(fixture.get("team_a"), {}).get("name", "unknown away team")
        event_id = fixture.get("event")
        event_name = events.get(event_id, {}).get("name", "gameweek TBC") if event_id else "gameweek TBC"
        status = "finished" if fixture.get("finished") else "upcoming or in progress"
        score = ""
        if fixture.get("team_h_score") is not None and fixture.get("team_a_score") is not None:
            score = f" Score: {fixture['team_h_score']}-{fixture['team_a_score']}."
        text = (
            f"FPL fixture: {home} vs {away}, {event_name}; kickoff: "
            f"{_display(fixture.get('kickoff_time'), 'not scheduled')}; status: {status}."
            f"{score} FPL data snapshot: {snapshot}."
        )
        add_document(
            f"fpl-fixture-{fixture['id']}",
            f"{API_BASE}/fixtures/ fixture {fixture['id']}",
            text,
            "fixture",
        )

    return documents


def ingest_documents() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in this project's .env before indexing FPL data.")

    bootstrap, fixtures = fetch_fpl_data()
    documents = build_documents(bootstrap, fixtures)
    if not documents:
        raise RuntimeError("The FPL API returned no indexable records.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )
    old_ids = vectorstore.get(include=["metadatas"])["ids"]
    document_ids = [
        hashlib.sha256(document.metadata["id"].encode()).hexdigest()
        for document in documents
    ]
    vectorstore.add_documents(documents, ids=document_ids)
    stale_ids = set(old_ids) - set(document_ids)
    if stale_ids:
        vectorstore.delete(ids=list(stale_ids))

    return len(documents)


def main() -> None:
    count = ingest_documents()
    print(f"Indexed {count} FPL records into {VECTOR_DB_DIR}.")


if __name__ == "__main__":
    main()
