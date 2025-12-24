from typing import Literal, TypedDict, List, Dict
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from finsight_rag.rag.rag_service import get_rag_service
from finsight_rag.agent.utils import dedupe_docs

# Get RAG service instance
rag_service = get_rag_service()

# --- State ---
class State(TypedDict, total=False):
    query: str # user question
    done: bool # whether finished retrieving/answering
    route_mode: Literal["rag", "general", "clarify"]
    # multi-hop
    hop: int
    max_hops: int
    subquestion: str # current subquestion to retrieve on
    notes: List[str] # accumulated notes
    answer: str
    docs_by_hop: List[List[Document]]
    last_docs: List[Document]

class RouteDecision(BaseModel):
    mode: Literal["rag", "general", "clarify"]

def route_node(state: State) -> State:
    llm = rag_service.llm
    q = (state.get("query") or "").strip()

    decision_dict = llm.with_structured_output(RouteDecision, method="json_schema").invoke(
        "Return ONLY valid JSON. No extra text.\n"
        "The JSON MUST contain exactly these keys:\n"
        '  - "mode": one of ["rag","general","clarify"]\n\n'
        'Template: {"mode":"rag"}\n\n'
        "You are a routing step for a financial QA assistant.\n"
        "Choose exactly one mode:\n"
        "- rag: requires company-specific financial documents\n"
        "- general: conceptual or educational, no documents needed\n"
        "- clarify: missing key info before retrieval is useful\n\n"
        "Rules:\n"
        "- If the question asks for specific figures, filings, quarters, or named companies -> rag\n"
        "- If the question is conceptual (definitions, frameworks, what metrics matter) -> general\n"
        "- If the question could become rag but lacks company, time period, or objective -> clarify\n"
        "- Do NOT answer the question.\n\n"
        f"User query:\n{q}\n"
    )
    
    decision = RouteDecision.model_validate(decision_dict)

    return {
        "route_mode": decision.mode,
    }

def route_next(state: State) -> str:
    return state["route_mode"]

def general_node(state: State) -> State:
    llm = rag_service.llm
    q = state["query"]

    answer = llm.invoke(
        "You are a helpful financial assistant.\n"
        "Answer the user's question using general knowledge only.\n"
        "Do NOT reference or imply the existence of internal documents.\n"
        "If helpful, ask a brief follow-up question at the end.\n\n"
        f"Question:\n{q}\n"
    ).content

    return {
        "answer": answer,
        "done": True,
    }

def clarify_node(state: State) -> State:
    llm = rag_service.llm
    q = state["query"]

    clarification = llm.invoke(
        "You are a financial assistant.\n"
        "The user's question is ambiguous.\n"
        "Ask ONE short clarification question that would enable document retrieval.\n\n"
        f"User question:\n{q}\n"
    ).content

    return {
        "answer": clarification,
        "done": True,
    }

# --- Plan next hop ---
class HopPlan(BaseModel):
    subquestion: str
    done: bool = False

def plan_multihop_rag_plan_node(state: State) -> State:
    llm = rag_service.llm
    hop = state.get("hop", 0)
    max_hops = state.get("max_hops", 3)

    if hop >= max_hops:
        return {"subquestion": state["query"], "done": True}

    notes = "\n".join(state.get("notes", []))
    plan_dict = llm.with_structured_output(HopPlan, method="json_schema").invoke(
        "You plan multi-hop retrieval over financial reports.\n"
        "Return ONLY valid JSON. No extra text.\n"
        "It MUST contain exactly these keys:\n"
        '  - "subquestion": string\n'
        '  - "done": boolean\n'
        f"User query: {state['query']}\n"
        f"Current notes:\n{notes}\n\n"
        "Propose the next best subquestion to retrieve more evidence.\n"
        "If enough evidence already, done=true.\n"
        "Do not set done=true unless you have evidence covering all parts of the query.\n"
    )
    plan = HopPlan.model_validate(plan_dict)

    return {"subquestion": plan.subquestion, "done": plan.done, "hop": hop + 1}

# --- Retrieve evidence using your existing retriever ---
def retrieve_node(state: State) -> State:
    print("Subquestion: ", state["subquestion"])
    docs = rag_service.retrieve(state["subquestion"])
    
    docs_by_hop = state.get("docs_by_hop", [])
    docs_by_hop.append(docs)

    return {"last_docs": docs, "docs_by_hop": docs_by_hop}

# --- Update notes (store what you learned this hop) ---
def notes_node(state: State) -> State:
    llm = rag_service.llm
    notes = state.get("notes", [])
    docs = state.get("last_docs", [])
    
    top_docs = docs[:3]

    excerpt = "\n\n".join(d.page_content[:800] for d in top_docs)
    summary = llm.invoke(
        "Write 2-4 bullet points of evidence from the excerpt. "
        "Preserve numbers, periods, and units.\n\n"
        f"Subquestion: {state['subquestion']}\n\n"
        f"Excerpt:\n{excerpt}"
    ).content

    notes.append(summary)
    return {"notes": notes}

def continue_or_end(state: State) -> str:
    if state.get("done"):
        return "final_rag"
    if state.get("hop", 0) >= state.get("max_hops", 3):
        return "final_rag"
    return "loop"

# --- Final answer ---
def final_rag_node(state: State) -> State:
    all_docs = []
    for hop_docs in state.get("docs_by_hop", []):
        all_docs.extend(hop_docs)

    all_docs = dedupe_docs(all_docs)

    answer, sources_str = rag_service.answer_from_docs(state["query"], all_docs)
    return {**state, "answer": answer + "\n\n---\nSources used:\n" + sources_str}

# --- Assemble graph ---
g = StateGraph(State)

g.add_node("route", route_node)
g.add_node("general", general_node)
g.add_node("clarify", clarify_node)
g.add_node("multihop_rag_plan", plan_multihop_rag_plan_node)
g.add_node("retrieve", retrieve_node)
g.add_node("notes_rag", notes_node)
g.add_node("final_rag", final_rag_node)

g.set_entry_point("route")
g.add_conditional_edges(
    "route",
    route_next,
    {
        "rag": "multihop_rag_plan",
        "general": "general",
        "clarify": "clarify",
    },
)
g.add_edge("multihop_rag_plan", "retrieve")
g.add_edge("retrieve", "notes_rag")
g.add_conditional_edges("notes_rag", continue_or_end, {"loop": "multihop_rag_plan", "final_rag": "final_rag"})
g.add_edge("final_rag", END)
g.add_edge("general", END)
g.add_edge("clarify", END)

app = g.compile()
