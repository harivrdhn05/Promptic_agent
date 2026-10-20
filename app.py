import os

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from agent import PromptOptimizer

load_dotenv()

st.set_page_config(page_title="Agentic Prompt Optimization Engine", page_icon="🧠", layout="wide")


FREE_MODELS = {
    "llama-3.1-8b-instant": "Fast, highest free limits (recommended)",
    "llama-3.3-70b-versatile": "Smarter, but lower free limits",
}


def get_api_key() -> str:
    return os.getenv("GROQ_API_KEY") or ""


def build_llm(model: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=get_api_key(),
        max_retries=3,
    )


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Groq model (free tier)", list(FREE_MODELS),
                         format_func=lambda m: f"{m}: {FREE_MODELS[m]}")
    custom_model = st.text_input("Or custom model name (optional)", placeholder="e.g. llama-3.1-8b-instant")
    if custom_model.strip():
        model = custom_model.strip()
    target = st.slider("Target score (stop when reached)", 5.0, 10.0, 8.5, 0.5)
    max_iter = st.slider("Max iterations", 1, 6, 3)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    generate_answers = st.checkbox("Compare answers (original vs optimized)", value=True)
    st.caption("Free tier is rate-limited. If you hit a limit, wait a minute "
               "or lower the iterations.")

# ---------------- Main ----------------
st.title("🧠 Agentic AI Prompt Optimization Engine")
st.caption("An agentic loop that evaluates, critiques and rewrites your prompt "
           "until it's comprehensive, context-rich and unambiguous.")

raw_prompt = st.text_area(
    "Your rough prompt",
    height=140,
    placeholder="e.g. write something about climate change for my class",
)

if st.button("🚀 Optimize prompt", type="primary", disabled=not raw_prompt.strip()):
    if not get_api_key():
        st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
        st.stop()

    try:
        llm = build_llm(model, temperature)
        optimizer = PromptOptimizer(llm, target_score=target, max_iterations=max_iter)

        with st.status("Running agentic optimization loop...", expanded=True) as status:
            def on_step(msg, iteration):
                st.write(msg)

            result = optimizer.run(raw_prompt, on_step=on_step)
            status.update(label=f"Done: {result.stopped_reason}", state="complete")

        answers = None
        if generate_answers:
            with st.spinner("Generating answers with both prompts..."):
                answers = (optimizer.answer(raw_prompt), optimizer.answer(result.final_prompt))

        st.session_state["result"] = result
        st.session_state["answers"] = answers
        st.session_state["raw"] = raw_prompt
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            st.error("Groq free-tier rate limit hit. Wait a minute and try again, "
                     "or reduce max iterations / turn off answer comparison.")
        else:
            st.error(f"Something went wrong: {e}")
        st.stop()

# ---------------- Results ----------------
result = st.session_state.get("result")
if result:
    first, best = result.iterations[0], max(result.iterations, key=lambda i: i.score)

    c1, c2, c3 = st.columns(3)
    c1.metric("Initial score", f"{first.score}/10")
    c2.metric("Final score", f"{best.score}/10", delta=round(best.score - first.score, 2))
    c3.metric("Iterations", len(result.iterations))

    st.subheader("✨ Optimized prompt")
    st.code(result.final_prompt, language="markdown", wrap_lines=True)

    st.subheader("📈 Score progression")
    st.line_chart({"Overall score": [i.score for i in result.iterations]})

    st.subheader("🔁 Iteration history")
    for it in result.iterations:
        ev = it.evaluation
        with st.expander(f"Iteration {it.number}: {it.score}/10",
                         expanded=(it is result.iterations[-1])):
            st.markdown("**Prompt at this stage**")
            st.code(it.prompt, language="markdown", wrap_lines=True)
            cols = st.columns(5)
            for col, (label, val) in zip(cols, [
                ("Clarity", ev.clarity), ("Specificity", ev.specificity),
                ("Context", ev.context), ("Constraints", ev.constraints),
                ("Format", ev.output_format)]):
                col.metric(label, val)
            a, b, c = st.columns(3)
            a.markdown("**✅ Strengths**\n" + "\n".join(f"- {s}" for s in ev.strengths))
            b.markdown("**⚠️ Weaknesses**\n" + "\n".join(f"- {s}" for s in ev.weaknesses))
            c.markdown("**💡 Suggestions**\n" + "\n".join(f"- {s}" for s in ev.suggestions))

    answers = st.session_state.get("answers")
    if answers:
        st.subheader("🆚 Response comparison")
        left, right = st.columns(2)
        left.markdown("**Original prompt response**")
        left.write(answers[0])
        right.markdown("**Optimized prompt response**")
        right.write(answers[1])
        