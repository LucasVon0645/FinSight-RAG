from typing import Literal, TypedDict, List
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from finsight_rag.rag.rag_service import get_rag_service
from finsight_rag.agent.utils import dedupe_docs, format_sources

# Get RAG service instance
rag_service = get_rag_service()

RouteModeType = Literal["single_hop_rag", "multihop_rag", "general", "clarify"]


# --- State ---
class State(TypedDict, total=False):
    query: str  # user question
    done: bool  # whether finished retrieving/answering
    route_mode: RouteModeType
    # multi-hop
    hop: int
    max_hops: int
    subquestion: str  # current subquestion to retrieve on
    notes: List[str]  # accumulated notes
    answer: str
    last_docs: List[Document]
    notes_src_docs: List[Document]


class RouteDecision(BaseModel):
    mode: RouteModeType


def route_node(state: State) -> State:
    llm = rag_service.llm
    q = (state.get("query") or "").strip()

    invoke_prompt = (
        "Return ONLY valid JSON. No extra text.\n"
        'Output format: {"mode": "<single_hop_rag | multihop_rag | general | clarify>"}\n\n'
        "You are a router for a financial QA system.\n\n"
        "Definitions:\n"
        "- single_hop_rag: one document, one company, one period, simple lookup.\n"
        "- multihop_rag: multiple documents/periods, comparisons, multiple metrics, trends, or explanations.\n"
        "- general: conceptual, no documents needed.\n"
        "- clarify: the query is not clear if is conceptual or document based.\n\n"
        "Rules (important):\n"
        "- Default to single_hop_rag.\n"
        "- Use multihop_rag ONLY if clearly required.\n"
        "- Simple questions like 'What was the revenue of Company X?' → single_hop_rag.\n"
        "- Comparisons, trends, multiple years/companies, or 'why/explain' → multihop_rag.\n\n"
        "Do NOT answer the question.\n\n"
        f"User query:\n{q}\n"
    )

    decision_dict = llm.with_structured_output(
        RouteDecision, method="json_schema"
    ).invoke(invoke_prompt)
    decision = RouteDecision.model_validate(decision_dict)

    out: State = {"route_mode": decision.mode}

    # Robust default init for multihop to prevent state leakage if state dict is reused.
    if decision.mode == "multihop_rag":
        out.update(
            {
                "hop": 0,
                "max_hops": state.get("max_hops", 3),
                "notes": [],
                "notes_src_docs": [],
                "last_docs": [],
                "done": False,
            }
        )

    return out


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
        "The user's question is ambiguous or incomplete.\n"
        "For example, missing company name, metric, or time period.\n"
        "Ask ONE short clarification question that would enable document retrieval.\n\n"
        f"User question:\n{q}\n"
    ).content

    return {
        "answer": clarification,
        "done": True,
    }


def single_hop_rag_node(state: State) -> State:
    answer, sources_str = rag_service.answer(question=state["query"], return_docs=False)
    return {
        **state,
        "answer": answer + "\n\n---\nSources used:\n" + sources_str,
        "done": True,
    }


# --- Plan next hop ---
class HopPlan(BaseModel):
    subquestion: str
    done: bool = False


def plan_multihop_rag_node(state: State) -> State:
    llm = rag_service.llm
    hop = state.get("hop", 0)
    max_hops = state.get("max_hops", 3)

    if hop >= max_hops:
        return {"done": True}

    notes = "\n".join(state.get("notes", []))
    plan_dict = llm.with_structured_output(HopPlan, method="json_schema").invoke(
        "You plan multi-hop retrieval over financial reports.\n"
        "Return ONLY valid JSON. No extra text.\n"
        "It MUST contain exactly these keys:\n"
        '  - "subquestion": string\n'
        '  - "done": boolean\n'
        f"User query: {state['query']}\n"
        f"Current notes:\n{notes}\n\n"
        "Propose the next best subquestion to retrieve more evidence if necessary.\n"
        "If enough evidence already, done=true.\n"
        "Do not set done=true unless you have evidence covering all parts of the query.\n"
    )
    plan = HopPlan.model_validate(plan_dict)
    
    if hop == 0:
        plan.done = False  # always do at least one hop
    
    if plan.done:
        return {"done": True}
    
    return {"subquestion": plan.subquestion, "hop": hop + 1}


# --- Retrieve evidence using your existing retriever ---
def retrieve_node(state: State) -> State:
    docs = rag_service.retrieve(state["subquestion"])

    return {"last_docs": docs}


# --- Update notes (store what you learned this hop) ---
def notes_node(state: State) -> State:
    llm = rag_service.llm
    notes = state.get("notes", [])
    docs = state.get("last_docs", [])
    notes_src_docs = state.get("notes_src_docs", [])

    top_docs = docs[:3]
    notes_src_docs.extend(top_docs)

    hop_sources = format_sources(top_docs)
    summary = llm.invoke(
        "Write 2-4 bullet points of evidence from the SOURCES.\n"
        "Rules:\n"
        "- Each bullet MUST end with a parenthetical citation that EXACTLY matches "
        "the source header text, e.g. (annual_report.pdf page=12 year=2023).\n"
        "- Do NOT invent new citation formats.\n"
        "- Preserve numbers, periods, and units.\n\n"
        f"Subquestion: {state['subquestion']}\n\n"
        f"SOURCES:\n{hop_sources}\n"
    ).content

    notes.append(summary)
    return {"notes": notes, "notes_src_docs": notes_src_docs}


def continue_or_end_multihop_rag(state: State) -> str:
    if state.get("done"):
        return "final_multihop_rag"
    return "retrieve"


# --- Final answer ---
def final_multihop_rag_node(state: State) -> State:
    notes = state.get("notes", [])
    notes_src_docs = dedupe_docs(state.get("notes_src_docs", []))

    sources_txt = format_sources(notes_src_docs)

    answer = rag_service.llm.invoke(
        "You are a financial assistant. Use the accumulated notes to answer the user's question.\n"
        "Keep citations from the notes in the final answer verbatim "
        "(e.g., (annual_report.pdf page=12 year=2023)).\n"
        "Do NOT invent new citation formats.\n\n"
        f"User query: {state['query']}\n"
        f"Notes:\n{'\n\n'.join(notes)}\n"
    ).content

    # Append sources for transparency/debugging.
    return {
        **state,
        "answer": answer + "\n\n---\nSources used:\n" + sources_txt,
    }


# --- Assemble graph ---
g = StateGraph(State)

g.add_node("route", route_node)
g.add_node("general", general_node)
g.add_node("clarify", clarify_node)
g.add_node("single_hop_rag", single_hop_rag_node)
g.add_node("plan_multihop_rag", plan_multihop_rag_node)
g.add_node("retrieve", retrieve_node)
g.add_node("notes_rag", notes_node)
g.add_node("final_multihop_rag", final_multihop_rag_node)

g.set_entry_point("route")
g.add_conditional_edges(
    "route",
    route_next,
    {
        "single_hop_rag": "single_hop_rag",
        "multihop_rag": "plan_multihop_rag",
        "general": "general",
        "clarify": "clarify",
    },
)


g.add_conditional_edges(
    "plan_multihop_rag",
    continue_or_end_multihop_rag
)
g.add_edge("retrieve", "notes_rag")
g.add_edge("notes_rag", "plan_multihop_rag")

g.add_edge("single_hop_rag", END)
g.add_edge("final_multihop_rag", END)
g.add_edge("general", END)
g.add_edge("clarify", END)

app = g.compile()
