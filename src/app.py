"""
app.py — Streamlit demo. This is the link that goes in the README.

    streamlit run src/app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from skade_agent import build_graph
from skade_agent.state import ClaimState

load_dotenv()

st.set_page_config(page_title="skade-agent", page_icon="🛡️", layout="wide")

st.title("🛡️ skade-agent")
st.caption(
    "Multi-step claims-intake agent built with LangGraph. "
    "Beskriv en skade på norsk (tekst eller stemme) — agenten henter ut "
    "strukturerte felter, klassifiserer, finner relevante vilkår, og "
    "drafter et svar."
)

with st.sidebar:
    st.markdown("### Hvordan det fungerer")
    st.markdown(
        "1. **Transcribe** — stemme → tekst (Whisper) hvis lyd\n"
        "2. **Extract** — strukturerte felter via Pydantic schema\n"
        "3. **Validate** — sjekk om vi har nok info\n"
        "4. **Classify** — bil / innbo / reise / person / annet\n"
        "5. **Lookup policy** — hent relevante vilkår\n"
        "6. **Draft response** — utkast til kundesvar med kilder"
    )
    st.markdown("---")
    st.markdown("**Stack:** LangGraph · OpenAI · Whisper · Pydantic")

mode = st.radio("Inndata", ["Tekst", "Stemme (lydfil)"], horizontal=True)

state: ClaimState | None = None
if mode == "Tekst":
    text = st.text_area(
        "Beskriv hva som skjedde",
        height=140,
        placeholder="F.eks. 'Jeg krasjet bilen min i går klokken 17 i Oslo. "
        "Jeg traff en parkert bil. Ingen personer ble skadet.'",
    )
    if st.button("Behandle saken", type="primary") and text.strip():
        state = ClaimState(raw_text=text)
else:
    audio = st.file_uploader("Last opp lydfil (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    if audio and st.button("Behandle saken", type="primary"):
        # Whisper needs a real file on disk
        with tempfile.NamedTemporaryFile(suffix=Path(audio.name).suffix, delete=False) as tmp:
            tmp.write(audio.getvalue())
            state = ClaimState(voice_file_path=tmp.name)

if state is not None:
    with st.status("Agenten arbeider...", expanded=False) as status:
        graph = build_graph()
        final_raw = graph.invoke(state)
        final = ClaimState.model_validate(final_raw)
        status.update(label="Ferdig", state="complete")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Utkast til kundesvar")
        st.write(final.response_draft or "_(ingen tekst generert)_")

        if final.next_steps:
            st.markdown("**Konkrete neste steg**")
            for s in final.next_steps:
                st.markdown(f"- {s}")

    with col_right:
        st.subheader("Strukturert sak (JSON)")
        st.json(final.model_dump(exclude_none=True, mode="json"))

        if final.policy_clauses:
            st.markdown("**Vilkår agenten brukte**")
            for c in final.policy_clauses:
                st.markdown(f"- **{c.source}** — {c.text}")
