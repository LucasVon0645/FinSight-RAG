from typing import Literal, TypedDict, List
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from finsight_rag.llms.llm_service import get_chat_llm_from_cfg
from finsight_rag.vector_store.vector_store_wrapper import VectorStoreWrapper
from finsight_rag.rag.rag_service import RAGService
from finsight_rag.agent.utils import dedupe_docs, format_sources
from finsight_rag.logging_config import logger, query_preview

MAPPING_ROUTE_MODE_TO_NODE = {
    "single_hop_rag": "single_hop_rag",
    "multihop_rag": "plan_multihop_rag",
    "general": "general",
    "clarify": "clarify",
}

RouteModeType = Literal["single_hop_rag", "multihop_rag", "general", "clarify"]

llm = get_chat_llm_from_cfg()
vector_store_wrapper = VectorStoreWrapper()  # load or create your Chroma vector store here
vector_store_retriever = vector_store_wrapper.get_retriever()

rag_service = RAGService(vector_store_retriever, llm)

# --- State ---
class State(TypedDict, total=False):
    query: str  # user question
    chat_history: List[BaseMessage]  # recent messages without sources
    done: bool  # whether finished retrieving/answering
    route_mode: RouteModeType
    route_reason: str
    # multi-hop
    hop: int
    max_hops: int
    subquestion: str  # current subquestion to retrieve on
    notes: List[str]  # accumulated notes
    subquestions: List[str] # all subquestions asked
    answer: str
    sources: str
    last_docs: List[Document]
    notes_src_docs: List[Document]


class RouteDecision(BaseModel):
    mode: RouteModeType
    reason: str


def format_chat_history(chat_history: List[BaseMessage] | None) -> str:
    """Format recent source-free messages for LLM prompts."""
    if not chat_history:
        return "No previous conversation."

    return "\n".join(
        f"- {'user' if isinstance(message, HumanMessage) else 'assistant'}: {message.content}"
        for message in chat_history
    ) or "No previous conversation."


def to_chat_messages(messages: List[dict[str, str]] | None) -> List[BaseMessage]:
    """Convert UI messages into LangChain messages, excluding source markup."""
    result: List[BaseMessage] = []
    for message in messages or []:
        content = message.get("content", "")
        if message.get("role") == "user":
            result.append(HumanMessage(content=content))
        elif message.get("role") == "assistant":
            result.append(AIMessage(content=content))
    return result


def route_node(state: State) -> State:
    q = (state.get("query") or "").strip()

    router_instructions = (
        "Return ONLY valid JSON. No extra text.\n"
        'Output format: {{"mode": "<single_hop_rag | multihop_rag | general | clarify>", "reason": "brief explanation"}}\n\n'
        "You are a router for a financial QA system.\n\n"
        "Definitions:\n"
        "- single_hop_rag: one document, one company, one period, simple lookup.\n"
        "- multihop_rag: multiple documents/periods, comparisons, multiple metrics, trends, or explanations.\n"
        "- general: conceptual, no documents needed.\n"
        "- clarify: the query is not clear if is conceptual or document based.\n\n"
        "Rules (important):\n"
        "- Resolve references such as 'this company', 'the company', 'they', 'it', and 'that period' using the previous conversation.\n"
        "- If the previous conversation identifies exactly one company or period, treat the reference as resolved and do not choose clarify.\n"
        "- A follow-up question inherits the company, period, and subject from the previous conversation unless the user explicitly changes them.\n"
        "- Choose clarify only when the reference has multiple plausible meanings or cannot be resolved from the conversation.\n"
        "- Default to single_hop_rag.\n"
        "- Use multihop_rag ONLY if clearly required.\n"
        "- Simple questions like 'What was the revenue of Company X?' → single_hop_rag.\n"
        "- Comparisons, trends, multiple years/companies, or 'why/explain' → multihop_rag.\n\n"
        "Decision requirements:\n"
        "- Return a brief reason that cites the evidence from the conversation or query.\n"
        "- If the query is a follow-up and the previous conversation contains exactly one plausible company, resolve the reference to that company.\n"
        "- For example, if the conversation mentions Embraer and the user asks 'How many aircraft did this company sell?', choose single_hop_rag, not clarify.\n"
        "Do NOT answer the question.\n"
    )

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", router_instructions),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "User query:\n{query}"),
    ])
    router_messages = router_prompt.format_messages(
        chat_history=state.get("chat_history") or [],
        query=q,
    )

    decision_dict = llm.with_structured_output(
        RouteDecision, method="json_schema"
    ).invoke(router_messages)
    decision = RouteDecision.model_validate(decision_dict)
    logger.info(
        "route_node decision mode={} reason={} query={}",
        decision.mode,
        decision.reason,
        query_preview(q),
    )

    out: State = {
        "route_mode": decision.mode,
        "route_reason": decision.reason,
    }

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
    route_mode = state.get("route_mode")
    return MAPPING_ROUTE_MODE_TO_NODE[route_mode]


def general_node(state: State) -> State:
    q = state["query"]

    answer = llm.invoke(
        "You are a helpful financial assistant.\n"
        "Answer the user's question using general knowledge only.\n"
        "Do NOT reference or imply the existence of internal documents.\n"
        "If helpful, ask a brief follow-up question at the end.\n\n"
        f"Previous conversation:\n{format_chat_history(state.get('chat_history'))}\n\n"
        f"Question:\n{q}\n"
    ).content

    return {
        "answer": answer,
        "done": True,
    }


def clarify_node(state: State) -> State:
    q = state["query"]

    clarification = llm.invoke(
        "You are a financial assistant.\n"
        "The user's question is ambiguous or incomplete.\n"
        "For example, missing company name, metric, or time period.\n"
        "Before asking for clarification, inspect the previous conversation and resolve references such as 'this company', 'the company', 'they', 'it', and 'that period'.\n"
        "If exactly one company or period is established in the conversation, use it and do not ask the user to repeat it.\n"
        "Ask ONE short clarification question that would enable document retrieval.\n\n"
        f"Previous conversation:\n{format_chat_history(state.get('chat_history'))}\n\n"
        f"User question:\n{q}\n"
    ).content

    return {
        "answer": clarification,
        "done": True,
    }


def single_hop_rag_node(state: State) -> State:
    answer, sources_str = rag_service.answer(
        question=state["query"],
        chat_history=state.get("chat_history"),
        return_docs=False,
    )
    return {
        **state,
        "answer": answer,
        "sources": sources_str,
        "done": True,
    }


# --- Plan next hop ---
class HopPlan(BaseModel):
    subquestion: str
    done: bool = False


def plan_multihop_rag_node(state: State) -> State:
    hop = state.get("hop", 0)
    max_hops = state.get("max_hops", 3)

    if hop >= max_hops:
        return {"done": True}

    notes = "\n".join(state.get("notes", []))
    previous_subquestions = "\n".join(state.get("subquestions", []))
    plan_dict = llm.with_structured_output(HopPlan, method="json_schema").invoke(
        "You are planning the next retrieval step for a financial QA system.\n"
        "Return ONLY valid JSON (no extra text).\n"
        'Output keys exactly: {"subquestion": string, "done": boolean}\n\n'
        f"Previous subquestions: {previous_subquestions}\n"
        f"Previous conversation:\n{format_chat_history(state.get('chat_history'))}\n"
        f"User query: {state['query']}\n"
        f"Current notes:\n{notes}\n\n"
        "Rules:\n"
        "- Propose ONLY ONE next subquestion.\n"
        "- Do NOT repeat any previous subquestion (unless you refine it to be more specific).\n"
        "- Set done=true ONLY if the notes already contain enough evidence to answer ALL parts of the user query.\n"
        "- If the query involves multiple entities/companies/periods/metrics, ask about ONE missing item at a time.\n"
    )
    plan = HopPlan.model_validate(plan_dict)
    
    if hop == 0:
        plan.done = False  # always do at least one hop
    
    if plan.done:
        return {"done": True}
    
    return {"subquestion": plan.subquestion, "hop": hop + 1}


# --- Retrieve evidence using your existing retriever ---
def retrieve_node(state: State) -> State:
    """Retrieve documents for current subquestion."""
    docs = rag_service.retrieve(state["subquestion"])

    return {"last_docs": docs}


# --- Update notes (store what you learned this hop) ---
def notes_node(state: State) -> State:
    """
    Extract factual evidence from retrieved documents 
    for current subquestion in multi-hop RAG.
    """
    notes = state.get("notes", [])
    subquestions = state.get("subquestions", [])
    docs = state.get("last_docs", [])
    notes_src_docs = state.get("notes_src_docs", [])

    top_docs = docs[:3]
    notes_src_docs.extend(top_docs)

    hop_sources = format_sources(top_docs)
    subquestion = state["subquestion"]
    summary = llm.invoke(
        "Extract factual evidence ONLY from the SOURCES.\n"
        "STRICT RULES:\n"
        "- Output ONLY 2-4 bullet points. No intro text.\n"
        "- Each bullet must be directly supported by the sources.\n"
        "- Each bullet MUST end with a parenthetical citation that EXACTLY matches a source header, "
        "e.g. (BTG-Annual-Report-2024.pdf page=64 year=2024).\n"
        "- DO NOT include apologies, explanations, or any 'missing data' claims.\n"
        "- If there is no relevant evidence in the sources, output EXACTLY: NONE\n"
        "- Preserve numbers, currency, and units.\n\n"
        f"Subquestion: {state['subquestion']}\n\n"
        f"SOURCES:\n{hop_sources}\n"
    ).content.strip()

    summary = summary.strip()
    if summary != "NONE" and summary:
        notes.append(summary)
    subquestions.append(subquestion)
    
    return {"notes": notes, "notes_src_docs": notes_src_docs, "subquestions": subquestions}


def continue_or_end_multihop_rag(state: State) -> str:
    """Decide whether to continue multi-hop RAG or finish."""
    if state.get("done"):
        return "final_multihop_rag"
    return "retrieve"


# --- Final answer ---
def final_multihop_rag_node(state: State) -> State:
    """Compile all notes obtained with multi-hop RAG into final answer."""
    notes = state.get("notes", [])
    notes_src_docs = dedupe_docs(state.get("notes_src_docs", []))

    sources_txt = format_sources(notes_src_docs)

    joined_notes = "\n\n".join(notes)

    answer = rag_service.llm.invoke(
        "You are a financial assistant. Use the accumulated notes to answer the user's question.\n"
        "Keep citations from the notes in the final answer, e.g., "
        "(annual_report.pdf page=12 year=2023).\n\n"
        f"User query: {state['query']}\n"
        f"Previous conversation:\n{format_chat_history(state.get('chat_history'))}\n"
        f"Notes:\n{joined_notes}\n"
    ).content

    return {
        **state,
        "answer": answer,
        "sources": sources_txt,
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
    ["single_hop_rag", "plan_multihop_rag", "general", "clarify"],
)


g.add_conditional_edges(
    "plan_multihop_rag",
    continue_or_end_multihop_rag,
    ['retrieve', 'final_multihop_rag'],
)
g.add_edge("retrieve", "notes_rag")
g.add_edge("notes_rag", "plan_multihop_rag")

g.add_edge("single_hop_rag", END)
g.add_edge("final_multihop_rag", END)
g.add_edge("general", END)
g.add_edge("clarify", END)

app = g.compile()
