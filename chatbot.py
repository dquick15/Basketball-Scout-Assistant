from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI

from ingest import build_records, dataframe_signature
from vector_store import ScoutVectorStore


INDEX_DIR = Path("scout_chatbot/.vector_cache")
INDEX_DIR.mkdir(exist_ok=True)


class BasketballScoutAssistant:
    def __init__(self, api_key: str, chat_model: str = "gpt-4.1-mini", embedding_model: str = "text-embedding-3-small") -> None:
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.vector_store = ScoutVectorStore(api_key=api_key, model=embedding_model)

    def ensure_index(self, df: pd.DataFrame) -> None:
        signature = dataframe_signature(df)
        index_path = INDEX_DIR / f"{signature}.faiss"
        metadata_path = INDEX_DIR / f"{signature}.pkl"

        if index_path.exists() and metadata_path.exists():
            self.vector_store.load(index_path, metadata_path)
            return

        records = build_records(df)
        self.vector_store.build(records)
        self.vector_store.save(index_path, metadata_path)

    def answer_question(self, question: str, df: pd.DataFrame) -> dict[str, object]:
        intents = self._structured_intents(question, df)
        retrieved_records = self.vector_store.similarity_search(question, top_k=6)
        response_text = self._generate_answer(question, intents, retrieved_records)
        return {
            "answer": response_text,
            "retrieved_records": retrieved_records,
            "insights": intents,
            "retrieval_backend": self.vector_store.backend,
        }

    def player_profile(self, player_name: str, df: pd.DataFrame) -> dict[str, object] | None:
        player_df = df[df["Player Name"] == player_name].sort_values(["Event Date", "Event Name"])
        if player_df.empty:
            return None

        latest = player_df.iloc[-1].to_dict()
        earliest = player_df.iloc[0].to_dict()
        improvement = float(latest["Overall Score"] - earliest["Overall Score"])
        return {
            "player": latest,
            "history": player_df,
            "improvement": round(improvement, 2),
        }

    def event_summary(self, event_name: str, df: pd.DataFrame) -> dict[str, object] | None:
        event_df = df[df["Event Name"] == event_name].copy()
        if event_df.empty:
            return None

        top_players = event_df.sort_values("Overall Score", ascending=False).head(5)
        upside_players = event_df.sort_values(["Growth Upside", "Overall Score"], ascending=False).head(5)
        return {
            "event_name": event_name,
            "player_count": int(event_df["Player Name"].nunique()),
            "average_score": round(float(event_df["Overall Score"].mean()), 2),
            "top_players": top_players,
            "upside_players": upside_players,
            "positions": event_df["Position"].value_counts().to_dict(),
        }

    def prospect_rankings(self, df: pd.DataFrame, metric: str = "Overall Score", top_k: int = 10) -> pd.DataFrame:
        columns = ["Player Name", "Team", "Grade", "Position", "Event Name", metric]
        return df.sort_values(metric, ascending=False)[columns].head(top_k).reset_index(drop=True)

    def similar_players(self, player_name: str) -> list[dict[str, object]]:
        return self.vector_store.similar_players(player_name, top_k=5)

    def _structured_intents(self, question: str, df: pd.DataFrame) -> dict[str, object]:
        lowered = question.lower()
        insights: dict[str, object] = {}

        if "improved" in lowered or "season" in lowered or "stock" in lowered:
            player_groups = []
            for player_name, player_df in df.groupby("Player Name"):
                ordered = player_df.sort_values(["Event Date", "Event Name"])
                improvement = float(ordered.iloc[-1]["Overall Score"] - ordered.iloc[0]["Overall Score"])
                player_groups.append(
                    {
                        "Player Name": player_name,
                        "Improvement": round(improvement, 2),
                        "Current Score": round(float(ordered.iloc[-1]["Overall Score"]), 2),
                        "Team": ordered.iloc[-1]["Team"],
                        "Grade": ordered.iloc[-1]["Grade"],
                        "Position": ordered.iloc[-1]["Position"],
                    }
                )
            insights["most_improved"] = pd.DataFrame(player_groups).sort_values(
                ["Improvement", "Current Score"], ascending=[False, False]
            ).head(10)

        if "highest upside" in lowered or "upside" in lowered:
            upside_df = df.dropna(subset=["Growth Upside"]).sort_values(["Growth Upside", "Overall Score"], ascending=False)
            insights["highest_upside"] = upside_df[
                ["Player Name", "Team", "Grade", "Position", "Event Name", "Growth Upside", "Overall Score"]
            ].head(10)

        if "top" in lowered or "best" in lowered:
            insights["top_players"] = df[
                ["Player Name", "Team", "Grade", "Position", "Event Name", "Overall Score"]
            ].sort_values("Overall Score", ascending=False).head(10)

        player_match = re.search(r"similar to ([a-zA-Z .'-]+)", question, flags=re.IGNORECASE)
        if player_match:
            player_name = player_match.group(1).strip()
            insights["similar_players"] = self.similar_players(player_name)

        return insights

    def _generate_answer(
        self,
        question: str,
        structured_insights: dict[str, object],
        retrieved_records: list[dict[str, object]],
    ) -> str:
        serialized_records = []
        for record in retrieved_records:
            serialized_records.append(
                {
                    "Player Name": record["Player Name"],
                    "Team": record["Team"],
                    "Grade": record["Grade"],
                    "Position": record["Position"],
                    "Event Name": record["Event Name"],
                    "Event Date": str(record["Event Date"]),
                    "Overall Score": record["Overall Score"],
                    "Growth Upside": record.get("Growth Upside"),
                    "Strengths": record["Strengths"],
                    "Development Areas": record["Development Areas"],
                    "Projection": record["Projection"],
                    "Similarity": round(float(record["similarity"]), 4),
                }
            )

        structured_sections: list[str] = []
        for label, value in structured_insights.items():
            if isinstance(value, pd.DataFrame):
                structured_sections.append(f"{label}:\n{value.to_string(index=False)}")
            else:
                structured_sections.append(f"{label}: {value}")

        system_prompt = (
            "You are a basketball scouting assistant. "
            "Answer only from the provided scouting context. "
            "Be concise, factual, and basketball-specific. "
            "If the question asks for rankings or comparisons, explain the basis clearly. "
            "If the data is insufficient, say so directly."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Structured insights:\n{'\n\n'.join(structured_sections) if structured_sections else 'None'}\n\n"
            f"Retrieved scouting records:\n{serialized_records}"
        )

        try:
            response = self.client.responses.create(
                model=self.chat_model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                ],
                max_output_tokens=700,
            )
            return response.output_text.strip()
        except Exception:
            return self._fallback_answer(question, structured_insights, retrieved_records)

    def _fallback_answer(
        self,
        question: str,
        structured_insights: dict[str, object],
        retrieved_records: list[dict[str, object]],
    ) -> str:
        lowered = question.lower()

        if "most_improved" in structured_insights:
            improved_df = structured_insights["most_improved"]
            top_rows = improved_df.head(5).to_dict(orient="records")
            lines = [
                f"{row['Player Name']} ({row['Team']}, {row['Position']}) improved by {row['Improvement']:+.2f} to {row['Current Score']:.2f}."
                for row in top_rows
            ]
            return "Most improved players in the current filter set:\n\n" + "\n".join(lines)

        if "highest_upside" in structured_insights and "upside" in lowered:
            upside_df = structured_insights["highest_upside"].head(5)
            lines = [
                f"{row['Player Name']} ({row['Team']}, {row['Position']}) has a Growth Upside of {row['Growth Upside']:.2f} with an Overall Score of {row['Overall Score']:.2f}."
                for row in upside_df.to_dict(orient="records")
            ]
            return "Highest-upside prospects in the current filter set:\n\n" + "\n".join(lines)

        if "top_players" in structured_insights and any(keyword in lowered for keyword in ["top", "best", "guard", "forward"]):
            top_df = structured_insights["top_players"].head(5)
            lines = [
                f"{row['Player Name']} ({row['Team']}, {row['Position']}) posted a {row['Overall Score']:.2f} Overall Score at {row['Event Name']}."
                for row in top_df.to_dict(orient="records")
            ]
            return "Top players in the current filter set:\n\n" + "\n".join(lines)

        if "similar_players" in structured_insights and structured_insights["similar_players"]:
            lines = [
                f"{record['Player Name']} ({record['Team']}, {record['Position']}) is a close similarity match with score {record['similarity']:.3f}."
                for record in structured_insights["similar_players"][:5]
            ]
            return "Similar-player results:\n\n" + "\n".join(lines)

        if retrieved_records:
            top_matches = retrieved_records[:5]
            lines = [
                f"{record['Player Name']} from {record['Team']} is a {record['Grade']} {record['Position']} with strengths in {record['Strengths']} and an Overall Score of {float(record['Overall Score']):.2f}."
                for record in top_matches
            ]
            return "Based on the closest scouting matches I found:\n\n" + "\n".join(lines)

        return "I could not find enough matching scouting context to answer that question from the current filters."


def resolve_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")
