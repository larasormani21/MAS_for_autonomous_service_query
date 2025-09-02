from langgraph.graph import StateGraph
from state import State
from agents.converter import ConverterAgent
from agents.executor import ExecutorAgent
from agents.feedback import FeedbackAgent
from agents.retreiver import RetrieverAgent
from agents.extractor import ExtractorAgent
from agents.indexer import Indexer

def print_node(node):
    print(f"======== {node} node ========")

def routing(state: State) -> str:
    if state.get("done"):
        return "end"

    if not state.get("retrieved"):
        if state.get("needs_reindex", False):
            print_node("index")
            return "index"
        print_node("retrieve")
        return "retrieve"

    idx = state.get("current_index", 0)
    files = state.get("candidate_files", []) or []
    if idx >= len(files):
        return "end"

    if not state.get("api_spec_yaml") or not state.get("system_message"):
        print_node("prepare")
        return "prepare"

    if not state.get("last_response"):
        print_node("request")
        return "request"

    if state.get("accepted") and not state.get("done"):
        print_node("extract")
        return "extract"
    print_node("feedback")
    return "feedback"


def build_multiagent_graph() -> StateGraph:
    g = StateGraph(State)
    g.add_node("index", Indexer().run)
    g.add_node("retrieve", RetrieverAgent().run)
    g.add_node("prepare", ConverterAgent().run)
    g.add_node("request", ExecutorAgent().run)
    g.add_node("feedback", FeedbackAgent().run)
    g.add_node("extract", ExtractorAgent().run)

    g.set_entry_point("retrieve")
    print_node("retrieve")

    for node in ["index", "retrieve", "prepare", "request", "feedback", "extract"]:
        g.add_conditional_edges(node, routing)

    return g


def run_with_multiagent(user_query: str) -> str:
    graph = build_multiagent_graph().compile()
    initial_state: State = {
        "user_query": user_query,
        "needs_reindex": False,
        "retrieved": False,
    }
    final_state = graph.invoke(initial_state)

    if final_state.get("done") and final_state.get("last_response"):
        print("Pipeline completed successfully.")
        return final_state["last_response"]

    raise RuntimeError(final_state.get("error") or "Pipeline did not complete successfully.")
