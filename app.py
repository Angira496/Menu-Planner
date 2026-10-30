"""Streamlit UI for the Indian occasion menu planner. Run: streamlit run app.py"""
import uuid

import streamlit as st
from langgraph.types import Command

from graph import QUESTIONS, build_graph, is_busy_error, menu_to_markdown

st.set_page_config(page_title="Utsav Menu Planner", page_icon="🍛")
st.title("🍛 Utsav Menu Planner")
st.caption("Culturally aware menu suggestions that fit your budget, built with LangChain + LangGraph")


def reset():
    st.session_state.graph = build_graph()          # a fresh graph + a fresh thread for every new event
    st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    st.session_state.result = None
    st.session_state.final = None                   # the accepted review payload
    st.session_state.submits = 0
    st.session_state.error = None


if "graph" not in st.session_state:
    reset()
ss = st.session_state


def advance(payload):
    """Run the graph until its next pause. If Gemini fails, keep everything and show a Try again button."""
    ss.error = None
    try:
        with st.spinner("Working on it... (writing the menu can take up to a minute)"):
            ss.result = ss.graph.invoke(payload, ss.config)
    except Exception as e:  # noqa: BLE001
        if is_busy_error(e):
            ss.error = ("Google's AI servers are busy right now (this is not a problem with your answers). "
                        "Everything you entered is saved. Please press **Try again** in a few seconds.")
        else:
            ss.error = f"Something went wrong: {e}"
    st.rerun()


def show_menu(menu: dict, info: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total budget", f"₹{info['budget']:,.0f}")
    c2.metric("Guests", f"{info['guests']}")
    c3.metric("Per guest", f"₹{info['per_plate']:,.0f}")
    st.caption(f"Budget level: **{info['budget_tier']}**. Low: up to ₹300 per guest, Medium: up to ₹700, High: above that.")
    for w in info.get("warnings", []):
        st.warning(w)
    st.subheader(menu["title"])
    courses = list(dict.fromkeys(d["course"] for d in menu["dishes"]))
    for c in courses:
        st.markdown(f"**{c}**")
        for d in (x for x in menu["dishes"] if x["course"] == c):
            icon = "🟢" if d["is_vegetarian"] else "🔴"
            st.markdown(f"{icon} **{d['name']}** — ₹{d['price']} per guest" + (f" — {d['note']}" if d["note"] else ""))
    st.success(f"Menu cost: ₹{menu['cost_per_guest']:,.0f} per guest × {info['guests']} guests = "
               f"₹{menu['total_cost']:,.0f} (your budget: ₹{info['budget']:,.0f})")
    st.info(f"💡 {menu['serving_tips']}")


# ---- something went wrong: keep the saved progress and let the user retry ----
if ss.error:
    st.error(ss.error)
    c1, c2 = st.columns(2)
    if c1.button("Try again", type="primary"):
        advance(None)          # None = continue from the last saved step
    if c2.button("Start over"):
        reset()
        st.rerun()
    st.stop()

# ---- not started ----
if ss.result is None and ss.final is None:
    st.write("Answer a few questions about your event and your budget, and I'll suggest a menu that "
             "respects your family's religion, culture and dietary needs, and fits your budget.")
    if st.button("Start planning", type="primary"):
        advance({"answers": {}})
    st.stop()

# ---- finished ----
if ss.final is not None:
    st.success("Menu confirmed! 🎉")
    show_menu(ss.final["menu"], ss.final)
    st.download_button("Download menu (.md)", menu_to_markdown(ss.final["menu"], ss.final), "menu.md")
    if st.button("Plan another event"):
        reset()
        st.rerun()
    st.stop()

# ---- in progress: show earlier answers (read from the graph's saved state) ----
answers = ss.graph.get_state(ss.config).values.get("answers", {})
for key, q, _ in QUESTIONS:
    if key in answers:
        st.chat_message("assistant").write(q)
        st.chat_message("user").write(answers[key])

payload = ss.result["__interrupt__"][0].value  # type: ignore

if payload["type"] == "question":
    st.chat_message("assistant").write(payload["question"])
    if payload.get("error"):
        st.warning(payload["error"])
    with st.form(key=f"q_{len(answers)}_{ss.submits}"):
        choice = st.selectbox("Choose", payload["options"]) if payload.get("options") else None
        typed = "" if choice else st.text_input("Your answer")
        if st.form_submit_button("Next"):
            answer = (typed or choice or "").strip()
            if not answer:
                st.warning("Please enter an answer.")
            else:
                ss.submits += 1
                advance(Command(resume=answer))
else:  # review
    show_menu(payload["menu"], payload)
    with st.form(key="review"):
        feedback = st.text_area("Want changes? e.g. 'add more starters', 'less spicy', 'remove paneer'")
        c1, c2 = st.columns(2)
        revise = c1.form_submit_button("Revise menu")
        accept = c2.form_submit_button("Accept menu", type="primary")
    if accept:
        ss.final = payload
        advance(Command(resume="ok"))
    elif revise and feedback.strip():
        advance(Command(resume=feedback.strip()))
    elif revise:
        st.warning("Type what you'd like changed first.")
