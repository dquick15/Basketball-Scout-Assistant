# Basketball Scout Assistant

Streamlit chatbot for basketball scouting databases using OpenAI plus FAISS-backed retrieval.

## Features

- Conversational scouting Q and A across players, events, rankings, and improvement trends.
- Retrieval-augmented generation with OpenAI responses grounded in scouting records.
- FAISS vector index for semantic retrieval of relevant player evaluations.
- Sidebar filters for Team, Grade, Position, and Event.
- Quick tools for player lookup, event summaries, prospect rankings, and similar-player search.
- Support for both CSV exports and the bundled AAU workbook format.

## File Structure

```text
scout_chatbot/
|-- ingest.py
|-- vector_store.py
|-- chatbot.py
|-- app.py
|-- requirements.txt
`-- README.md
```

## Expected Fields

- Player Name
- Team
- Grade
- Position
- Strengths
- Development Areas
- Projection
- Event Name
- Event Date
- Overall Score

Optional:

- Growth Upside

## Setup

1. Install dependencies:

```bash
pip install -r scout_chatbot/requirements.txt
```

2. Set your OpenAI API key.
3. Run the app:

```bash
streamlit run scout_chatbot/app.py
```

## Notes

- The app normalizes the bundled `AAU_Scouting_System.xlsx` workbook automatically when no CSV is uploaded.
- FAISS index files are cached under `scout_chatbot/.vector_cache/` using a signature of the active dataset.
- Similar-player search is based on nearest neighbors in embedding space.
