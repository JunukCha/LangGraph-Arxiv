import os
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
import streamlit as st
import time

# Load environment variables
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
tools = [arxiv]
llm = llm.bind_tools(tools=tools)

class State(TypedDict):
    query: str
    content: str
    summary: List[str]
    evaluation: List[str]
    passed_check: bool
    iterations: int

# Node 1: Fetch and summarize content
def step_summarize(state: State) -> dict:
    arxiv_result = arxiv.run(state["query"])
    prompt = f"""
You are a helpful AI. Here is information from ArXiv:

{arxiv_result}

Summarize the key ideas in 3-5 lines.
"""
    response = llm.invoke(prompt)
    new_summary = response.content.strip()
    updated_summary = state.get("summary", []) + [new_summary]
    return {"content": arxiv_result[:5000], "summary": updated_summary, "iterations": state.get("iterations", 0) + 1}

# Node 2: Evaluate summary
def step_evaluate_summary(state: State) -> dict:
    latest_summary = state["summary"][-1] if state["summary"] else ""
    prompt = f"""
Evaluate the following summary for the query: "{state['query']}"

Summary:
{latest_summary}

Evaluate based on:
1. Accuracy
2. Relevance
3. Clarity
4. Completeness

Score each from 1 to 5 with short feedback.
"""
    response = llm.invoke(prompt)
    evaluation_text = response.content.strip()
    updated_evaluation = state.get("evaluation", []) + [evaluation_text]
    return {"evaluation": updated_evaluation}

# Node 3: Check pass/fail
def step_check_pass(state: State) -> dict:
    if state.get("iterations", 0) >= 3:
        return {"passed_check": True}
    latest_eval = state["evaluation"][-1] if state["evaluation"] else ""
    prompt = f"""
Here is an evaluation of a summary:
{latest_eval}

Does this evaluation indicate PASS or FAIL?
Respond with only one word: PASS or FAIL.
"""
    response = llm.invoke(prompt)
    passed = "pass" in response.content.strip().lower()
    return {"passed_check": passed}

# Final step
def step_finalize(state: State) -> dict:
    return state

# Conditional routing
def decision_router(state: State) -> str:
    if state.get("passed_check") or state.get("iterations", 0) >= 3:
        return "step_finalize"
    return "step_summarize"

# Graph definition
graph_builder = StateGraph(State)
graph_builder.add_node("step_summarize", step_summarize)
graph_builder.add_node("step_evaluate_summary", step_evaluate_summary)
graph_builder.add_node("step_check_pass", step_check_pass)
graph_builder.add_node("step_finalize", step_finalize)

graph_builder.set_entry_point("step_summarize")
graph_builder.add_edge("step_summarize", "step_evaluate_summary")
graph_builder.add_edge("step_evaluate_summary", "step_check_pass")
graph_builder.add_conditional_edges("step_check_pass", decision_router, {
    "step_summarize": "step_summarize",
    "step_finalize": "step_finalize"
})
graph_builder.set_finish_point("step_finalize")

compiled_graph = graph_builder.compile()

def run(query: str) -> State:
    init_state: State = {
        "query": query,
        "content": "",
        "summary": [],
        "evaluation": [],
        "passed_check": False,
        "iterations": 0
    }
    state = init_state
    while True:
        state = compiled_graph.invoke(state)
        yield state
        if state.get("passed_check") or state.get("iterations", 0) >= 3:
            break

# Streamlit UI
st.title("LangGraph")
query = st.text_input("Enter your research query:")
run_button_placeholder = st.empty()
run_button = run_button_placeholder.button("Evaluate")

if run_button and query:
    with st.spinner("Running LLM Evaluation..."):
        result = None
        for i, state in enumerate(run(query)):
            st.subheader("Summary + Evaluation Log")
            for j, (s, e) in enumerate(zip(state["summary"], state["evaluation"])):
                st.markdown(f"**Step {j+1} Summary:**\n{s}")
                st.markdown(f"**Step {j+1} Evaluation:**\n{e}")
            st.subheader("Passed")
            st.text("✅ PASS" if state["passed_check"] else "❌ FAIL")
            time.sleep(0.3)
    st.success("Evaluation completed.")
    run_button_placeholder.empty()