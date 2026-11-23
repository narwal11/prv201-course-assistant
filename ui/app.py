"""Streamlit chat UI.   Run:  streamlit run ui/app.py   (backend must be running)"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PRV201 Course Assistant", page_icon="📘")
st.title("PRV201 Course Assistant")
st.caption("Answers come only from PRV201 materials, with lecture/page citations. "
           "Code runs in an isolated sandbox. Hints only for graded work.")

with st.sidebar:
    lec = st.number_input("Limit to lecture (0 = all)", min_value=0, max_value=30, value=0, step=1)
    show_debug = st.checkbox("Show retrieved chunks and trace", value=False)
    if st.button("Clear chat"):
        st.session_state.history = []

st.session_state.setdefault("history", [])


def render(res: dict) -> None:
    st.markdown(res["answer"])
    for t in res.get("tool_calls", []):
        r = t["result"]
        status = "timed out" if r["timed_out"] else f"exit {r['exit_code']}"
        with st.expander(f"run_python ({status}, {r['duration_ms']} ms)"):
            st.code(t["code"], language="python")
            if r["stdout"]:
                st.text(r["stdout"])
            if r["stderr"]:
                st.error(r["stderr"])
    if res.get("citations"):
        st.caption("Sources: " + "; ".join(f"{c['label']} ({c['source']})" for c in res["citations"]))
    meta = f"{res['latency_ms']} ms" + (" · cached" if res.get("cached") else "")
    if res.get("flags"):
        meta += " · flags: " + ", ".join(res["flags"])
    st.caption(meta)
    if show_debug:
        with st.expander("Retrieved chunks"):
            for r in res["retrieved"]:
                st.markdown(f"**{r['tag']}** {r['label']} — sim {r['similarity']:.3f}")
                st.text(r["text"][:600])


for role, payload in st.session_state.history:
    with st.chat_message(role):
        render(payload) if role == "assistant" else st.markdown(payload)

if q := st.chat_input("Ask about PRV201…"):
    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(f"{API_URL}/ask", json={"question": q, "lecture": lec or None}, timeout=90)
                resp.raise_for_status()
                res = resp.json()
            except requests.RequestException as e:
                st.error(f"Backend error: {e}")
                st.stop()
        render(res)
    st.session_state.history.append(("assistant", res))
