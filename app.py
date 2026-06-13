from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from chatbot import BasketballScoutAssistant, resolve_api_key
from ingest import apply_filters, load_chatbot_data


st.set_page_config(page_title="Basketball Scout Assistant", layout="wide")


@st.cache_resource(show_spinner=False)
def get_assistant(api_key: str) -> BasketballScoutAssistant:
    return BasketballScoutAssistant(api_key=api_key)


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    teams = st.sidebar.multiselect("Team", options=sorted(df["Team"].unique()))
    grades = st.sidebar.multiselect("Grade", options=sorted(df["Grade"].unique()))
    positions = st.sidebar.multiselect("Position", options=sorted(df["Position"].unique()))
    events = st.sidebar.multiselect("Event", options=sorted(df["Event Name"].unique()))
    return apply_filters(df, teams=teams, grades=grades, positions=positions, events=events)


def render_quick_tools(assistant: BasketballScoutAssistant, df: pd.DataFrame) -> None:
    tab_one, tab_two, tab_three = st.tabs(["Player Lookup", "Event Summary", "Prospect Rankings"])

    with tab_one:
        player_name = st.selectbox("Player profile", options=sorted(df["Player Name"].unique()))
        profile = assistant.player_profile(player_name, df)
        if profile is not None:
            player = profile["player"]
            stats_one, stats_two, stats_three, stats_four = st.columns(4)
            stats_one.metric("Team", player["Team"])
            stats_two.metric("Grade", player["Grade"])
            stats_three.metric("Position", player["Position"])
            stats_four.metric("Score Change", f"{profile['improvement']:+.2f}")
            st.markdown(f"**Strengths**\n\n{player['Strengths']}")
            st.markdown(f"**Development Areas**\n\n{player['Development Areas']}")
            st.markdown(f"**Projection**\n\n{player['Projection']}")

            trend_fig = px.line(
                profile["history"],
                x="Event Date",
                y="Overall Score",
                markers=True,
                title=f"Overall Score Trend: {player_name}",
                hover_data=["Event Name"],
            )
            trend_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(trend_fig, width="stretch")

    with tab_two:
        event_name = st.selectbox("Event summary", options=sorted(df["Event Name"].unique()))
        summary = assistant.event_summary(event_name, df)
        if summary is not None:
            stat_one, stat_two = st.columns(2)
            stat_one.metric("Players Evaluated", summary["player_count"])
            stat_two.metric("Average Overall Score", f"{summary['average_score']:.2f}")
            st.markdown("**Top performers**")
            st.dataframe(summary["top_players"][["Player Name", "Team", "Position", "Overall Score"]], hide_index=True, width="stretch")
            st.markdown("**Most promising prospects**")
            st.dataframe(summary["upside_players"][["Player Name", "Team", "Position", "Growth Upside", "Overall Score"]], hide_index=True, width="stretch")

    with tab_three:
        ranking_metric = st.selectbox("Ranking metric", options=["Overall Score", "Growth Upside"])
        rankings = assistant.prospect_rankings(df, metric=ranking_metric, top_k=15)
        st.dataframe(rankings, hide_index=True, width="stretch")


def render_similar_players(assistant: BasketballScoutAssistant, df: pd.DataFrame) -> None:
    st.subheader("Similar Player Search")
    selected_player = st.selectbox("Find similar players", options=sorted(df["Player Name"].unique()), key="similar-player")
    if st.button("Find Similar Players", width="stretch"):
        matches = assistant.similar_players(selected_player)
        if matches:
            match_df = pd.DataFrame(matches)[
                ["Player Name", "Team", "Grade", "Position", "Event Name", "Overall Score", "similarity"]
            ]
            st.dataframe(match_df, hide_index=True, width="stretch")
        else:
            st.info("No similar players were found for the selected player.")


def render_chat(assistant: BasketballScoutAssistant, df: pd.DataFrame) -> None:
    st.subheader("Scout Chat")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask about players, events, rankings, upside, or improvement trends.")
    if not prompt:
        return

    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching scouting records..."):
            result = assistant.answer_question(prompt, df)
        st.write(result["answer"])

        with st.expander("Retrieved scouting context"):
            retrieved_df = pd.DataFrame(result["retrieved_records"])
            if not retrieved_df.empty:
                st.dataframe(
                    retrieved_df[
                        [
                            "Player Name",
                            "Team",
                            "Grade",
                            "Position",
                            "Event Name",
                            "Overall Score",
                            "similarity",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )

    st.session_state["chat_history"].append({"role": "assistant", "content": result["answer"]})


def main() -> None:
    st.title("Basketball Scout Assistant")
    st.write("Ask scouting questions across players, events, rankings, and development trends using a FAISS-backed scouting knowledge base.")

    try:
        df, source_name = load_chatbot_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    filtered_df = render_filters(df)
    if filtered_df.empty:
        st.warning("No scouting records match the selected filters.")
        st.stop()

    st.sidebar.success(f"Loaded data source: {source_name}")
    api_key = resolve_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is not configured. Set it before using the chatbot.")
        st.stop()

    assistant = get_assistant(api_key)
    with st.spinner("Building vector knowledge base..."):
        assistant.ensure_index(filtered_df)

    if assistant.vector_store.backend != "openai":
        st.warning(
            "OpenAI embeddings were unavailable for this session, so retrieval is using a local FAISS hash embedding fallback. "
            "Chat answers remain grounded in the data, but semantic quality may be lower until API quota is available."
        )

    render_quick_tools(assistant, filtered_df)
    render_similar_players(assistant, filtered_df)
    render_chat(assistant, filtered_df)


if __name__ == "__main__":
    main()
