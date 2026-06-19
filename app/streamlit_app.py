from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.orchestrator import respond
from src.config import config
from src.ingest.pipeline import index_count, index_has_data, ingest
from src.tools.access_control import all_users


st.set_page_config(page_title="Clinical RAG Prototype", layout="wide")


def _render_ingestion_tab() -> None:
    st.subheader("Data / Ingestion")
    st.caption(f"Azure production | Index: {config.index_name}")

    try:
        has_data = index_has_data()
        current_count = index_count() if has_data else 0
    except RuntimeError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not inspect Azure Search index: {exc}")
        return

    if not has_data:
        st.info(
            f"No data is registered for index `{config.index_name}`. "
            "Click Upload & Ingest to process the de-identified histories."
        )
    else:
        st.success(f"Index ready: {current_count} chunks")

    if st.button("Upload & Ingest", type="primary"):
        lines: list[str] = []
        progress_box = st.empty()

        def progress(event: str, payload: dict) -> None:
            if event == "start":
                lines.append(f"Documents found: {payload['documents_found']}")
            elif event == "document":
                lines.append(
                    "- {source_doc}: {chunks_created} chunks, {uploaded} uploaded, {skipped} skipped".format(
                        **payload
                    )
                )
            elif event == "complete":
                lines.append(f"Index ready: {payload['total_index_chunks']} chunks")
            progress_box.markdown("\n".join(lines))

        with st.spinner("Processing documents"):
            try:
                stats = ingest(progress=progress)
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
                return

        st.success(f"Index ready: {stats.total_index_chunks} chunks")
        st.dataframe(
            [
                {
                    "source_doc": doc.source_doc,
                    "patient_id": doc.patient_id,
                    "chunks": doc.chunks_created,
                    "uploaded": doc.uploaded,
                    "skipped": doc.skipped,
                }
                for doc in stats.per_document
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_chat_tab() -> None:
    st.subheader("Clinical Chat")

    users = all_users()
    selected_user = st.selectbox(
        "Acting as",
        users,
        index=0,
        key="acting_user",
        help="This user ID is injected server-side into the Azure Search ACL filter.",
    )

    if "chat_by_user" not in st.session_state:
        st.session_state.chat_by_user = {}
    history = st.session_state.chat_by_user.setdefault(selected_user, [])

    status_col, action_col = st.columns([2, 1], vertical_alignment="center")
    with status_col:
        st.caption(f"Current user: `{selected_user}`")
        try:
            if not index_has_data():
                st.warning("No indexed data yet. Use Data / Ingestion first.")
        except Exception as exc:
            st.warning(f"Azure Search index status unavailable: {exc}")
    with action_col:
        if st.button("Clear chat", use_container_width=True):
            history.clear()
            st.rerun()

    st.divider()

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_meta(message.get("meta"))

    prompt = st.chat_input("Ask a clinical records question")
    if not prompt:
        return

    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking access and records"):
            try:
                output = respond(prompt, user_id=selected_user)
            except Exception as exc:
                output = {
                    "answer": f"Request failed: {exc}",
                    "meta": {"path": "error", "type": "n/a"},
                }
        st.markdown(output["answer"])
        _render_meta(output["meta"])

    history.append(
        {"role": "assistant", "content": output["answer"], "meta": output["meta"]}
    )


def _render_meta(meta: dict | None) -> None:
    if not meta:
        return
    st.caption(f"path: `{meta.get('path')}` | query type: `{meta.get('type', 'n/a')}`")


tab_ingest, tab_chat = st.tabs(["Data / Ingestion", "Chatbot"])
with tab_ingest:
    _render_ingestion_tab()
with tab_chat:
    _render_chat_tab()
