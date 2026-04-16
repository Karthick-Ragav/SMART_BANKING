# src/api/v1/agent/agent.py

from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import cohere
import json
from langgraph.graph import StateGraph, START, END

# Tools
from src.api.v1.tools.vector_search_tool import vector_tool
from src.api.v1.tools.fts_tool import fts_tool
from src.api.v1.tools.hybrid_search_tool import hybrid_tool
from src.api.v1.tools.nl2sql_tool import nl2sql_tool

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

MAX_ITERATIONS = 3

IRRELEVANT_MESSAGE = (
    "This question is irrelevant to the available knowledge base. "
    "Please ask a relevant question."
)

# =========================
# STATE
# =========================
class AgentState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str
    iteration: int
    validate: bool
    tool_used: str
    generated_sql: str
    sql_result: str


# =========================
# SAFE TEXT EXTRACTOR
# =========================
def extract_text(response):
    try:
        if isinstance(response.content, list):
            return response.content[0]["text"]
        return str(response.content)
    except:
        return str(response)


# =========================
# AGENT NODE (UPDATED)
# =========================
def agent_node(state: AgentState) -> AgentState:

    query = state["query"]

    prompt = f"""
You are a strict routing agent.

Your job:
1. Decide which system should answer the query
2. Select the correct tool

SYSTEMS:

--- RAG (Banking Knowledge Base) ---
Contains:
- Home loans, fixed deposits, credit cards, personal loans
- Interest rates, eligibility, charges, policies

Tools:
- vector → semantic understanding
- fts → exact keyword lookup
- hybrid → combination

--- NL2SQL (Database) ---
Contains:
- accounts
- transactions
- loan_accounts
- fixed_deposits
- credit_cards
- card_transactions

Use nl2sql ONLY for:
- account information
- transaction history and spending analysis
- loan details (EMI, outstanding, interest rate)
- fixed deposit details (maturity, interest, status)
- credit card details (limits, dues, transactions)
- aggregations (total spend, counts, summaries)

Examples:
- "show transactions for account 1345367"
- "total spend last month"
- "outstanding loan amount"
- "next EMI date"
- "credit card due amount"
- "FD maturity details"

--- HYBRID QUERY ---
If query contains BOTH:
- banking/document question AND
- product/database query
THEN return:
{{"tool": "hybrid_query"}}

--- IRRELEVANT ---
If query is unrelated to BOTH systems

-------------------------------------

DECISION RULES:

1. Banking/doc questions → RAG tools
2. Banking/database queries (accounts, transactions, loans, FD, credit cards) → nl2sql
3. Mixed (RAG + DB) → hybrid_query
4. Otherwise → irrelevant

-------------------------------------

RAG TOOL SELECTION:

- vector → explanations
- fts → exact keywords
- hybrid → mixed queries

-------------------------------------

OUTPUT FORMAT (STRICT):

Return ONLY valid JSON.
No explanation. No extra text.

Format:
{{"tool": "vector"}}

Allowed values:
"vector"
"fts"
"hybrid"
"nl2sql"
"hybrid_query"
"irrelevant"

INVALID:
- Sure! {{"tool": "vector"}}
- ```json ... ```
- The answer is vector

-------------------------------------

Query:
"{query}"
"""

    response = extract_text(llm.invoke(prompt)).strip()
    print(f"[Agent Raw Output] {response}")

    tool = "vector"
    try:
        import re
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            tool = parsed.get("tool", "vector").lower()
    except Exception:
        print("[Agent] JSON parsing failed, defaulting to vector")

    if tool == "irrelevant":
        print("[Router] Irrelevant query detected")
        state["tool_used"] = "irrelevant"
        state["answer"] = IRRELEVANT_MESSAGE
        state["retrieved_docs"] = []
        state["reranked_docs"] = []
        return state

    if tool not in ["vector", "fts", "hybrid", "nl2sql", "hybrid_query"]:
        tool = "vector"

    state["tool_used"] = tool
    print(f"[Router] Selected tool → {tool}")

    return state


# =========================
# ROUTING
# =========================
def route_after_agent(state: AgentState):

    if state["tool_used"] == "irrelevant":
        return "generate"

    if state["tool_used"] == "nl2sql":
        return "nl2sql"

    if state["tool_used"] == "hybrid_query":
        return "hybrid_query"

    return "retrieve"


# =========================
# NL2SQL NODE
# =========================
def nl2sql_node(state: AgentState) -> AgentState:

    print("[NL2SQL] Executing...")

    result = nl2sql_tool.invoke(state["query"])

    state["answer"] = result["answer"]
    state["generated_sql"] = result["sql_query"]
    state["sql_result"] = result["sql_result"]

    print("[NL2SQL] Done")
    return state


# =========================
#  HYBRID QUERY NODE 
# =========================
def hybrid_query_node(state: AgentState) -> AgentState:

    print("[Hybrid Query] Executing with query splitting...")

    query = state["query"]

    # -------------------------
    # Step 1: Split Query
    # -------------------------
    split_prompt = f"""
You are a query decomposition agent.

Split the query into TWO parts:
1. Document-related (banking / RAG)
2. Database-related (products/orders)

STRICT RULES:
- Return ONLY valid JSON
- No explanation
- No extra text

Format:
{{
  "rag_query": "...",
  "sql_query": "..."
}}

If one part is missing, return empty string "" for that field.

Query:
"{query}"
"""

    split_response = extract_text(llm.invoke(split_prompt)).strip()
    print(f"[Hybrid Split Raw] {split_response}")

    rag_query = query
    sql_query = query

    # Safe parsing
    try:
        import re
        match = re.search(r"\{.*\}", split_response, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            rag_query = parsed.get("rag_query") or query
            sql_query = parsed.get("sql_query") or query
    except Exception:
        print("[Hybrid Split] Failed → fallback to original query")

    print(f"[Hybrid Split] RAG Query: {rag_query}")
    print(f"[Hybrid Split] SQL Query: {sql_query}")

    # -------------------------
    # Step 2: RAG Retrieval
    # -------------------------
    # -------------------------

    docs = []
    reranked_docs = []
    context = ""

    if rag_query.strip():

        # Step 2.1 Retrieve
        docs = hybrid_tool.invoke(rag_query)

        # Step 2.2 Rerank
        if docs:
            co_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

            response = co_client.rerank(
                model="rerank-english-v3.0",
                query=rag_query,
                documents=[d.page_content for d in docs],
                top_n=5
            )

            reranked_docs = [docs[r.index] for r in response.results]

        # Step 2.3 Context
        context = "\n\n".join([d.page_content for d in reranked_docs]) if reranked_docs else ""

    # -------------------------
    # Step 3: SQL Execution
    # -------------------------
    sql_result = {}

    if sql_query.strip():
        try:
            sql_result = nl2sql_tool.invoke(sql_query)
        except Exception as e:
            print(f"[Hybrid SQL Error] {e}")
            sql_result = {"sql_result": ""}

    # -------------------------
    # Step 4: Combine Answer
    # -------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a precise assistant.

Rules:
- Answer directly
- No phrases like "Based on the context"
- Combine both document and database answers
- If one part is missing, answer only the available part
- If both missing, say "No data found"
"""),
        ("human", """
Document Context:
{context}

SQL Result:
{sql}

Question:
{query}

Answer:
""")
    ])

    chain = prompt | llm

    result = chain.invoke({
        "context": context,
        "sql": sql_result.get("sql_result", ""),
        "query": query
    })

    state["answer"] = extract_text(result)

    print("[Hybrid Query] Done")

    return state

# =========================
# RETRIEVE NODE
# =========================
def retrieve_node(state: AgentState) -> AgentState:

    tool = state["tool_used"]

    print(f"[Retrieve] Using tool: {tool}")

    if tool == "vector":
        docs = vector_tool.invoke(state["query"])
    elif tool == "fts":
        docs = fts_tool.invoke(state["query"])
    else:
        docs = hybrid_tool.invoke(state["query"])

    if not docs:
        print("[Retrieve] No results → switching to HYBRID")
        docs = hybrid_tool.invoke(state["query"])
        state["tool_used"] = "hybrid"

    state["retrieved_docs"] = docs

    print(f"[Retrieve] Retrieved {len(docs)} docs")
    return state


# =========================
# RERANK NODE
# =========================
def rerank_node(state: AgentState) -> AgentState:

    if not state["retrieved_docs"]:
        state["reranked_docs"] = []
        return state

    co_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    response = co_client.rerank(
        model="rerank-english-v3.0",
        query=state["query"],
        documents=[d.page_content for d in state["retrieved_docs"]],
        top_n=6
    )

    state["reranked_docs"] = [
        state["retrieved_docs"][r.index] for r in response.results
    ]

    print("[Rerank] Done")
    return state


# =========================
# VALIDATE NODE
# =========================
def validate_node(state: AgentState) -> AgentState:

    if state["iteration"] >= MAX_ITERATIONS or not state["reranked_docs"]:
        state["validate"] = bool(state["reranked_docs"])
        return state

    context = "\n\n".join([d.page_content for d in state["reranked_docs"]])

    prompt = f"""
Are these chunks directly relevant to the query?

Query: {state['query']}

Answer ONLY YES or NO.
"""

    decision = extract_text(llm.invoke(prompt)).lower()
    state["validate"] = "yes" in decision

    print(f"[Validate] {state['validate']}")
    return state


def route_after_validate(state: AgentState):
    return "generate" if state["validate"] else "rephrase"


# =========================
# REPHRASE NODE
# =========================
def rephrase_node(state: AgentState) -> AgentState:

    prompt = f"""
You are a query rewriter.

Rewrite the query for better retrieval.

Return ONLY the rewritten query.
No explanation.
No prefixes.
No quotes.

Original Query:
{state['query']}
"""

    new_query = extract_text(llm.invoke(prompt)).strip()

    print(f"[Rephrase OLD] {state['query']}")
    print(f"[Rephrase NEW] {new_query}")

    state["query"] = new_query
    state["iteration"] += 1
    state["retrieved_docs"] = []
    state["reranked_docs"] = []

    return state


# =========================
# GENERATE NODE
# =========================
def generate_node(state: AgentState) -> AgentState:

    if state["tool_used"] in ["nl2sql", "irrelevant", "hybrid_query"]:
        return state

    NO_DATA_MESSAGE = "No data found for the given query."

    if not state["reranked_docs"]:
        state["answer"] = NO_DATA_MESSAGE
        return state

    context = "\n\n".join([d.page_content for d in state["reranked_docs"]])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a precise assistant.

Rules:
- Answer directly
- No explanation phrases
- If missing data → say "No data found"
"""),
        ("human", """
Context:
{context}

Question:
{query}

Answer:
""")
    ])

    chain = prompt | llm
    result = chain.invoke({"context": context, "query": state["query"]})

    state["answer"] = extract_text(result)

    print("[Generate] Done")
    return state


# =========================
# WORKFLOW
# =========================
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("nl2sql", nl2sql_node)
workflow.add_node("hybrid_query", hybrid_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("validate", validate_node)
workflow.add_node("rephrase", rephrase_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("agent")

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "nl2sql": "nl2sql",
        "hybrid_query": "hybrid_query",
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

workflow.add_edge("nl2sql", "generate")
workflow.add_edge("hybrid_query", "generate")

workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "validate")

workflow.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "generate": "generate",
        "rephrase": "rephrase"
    }
)

workflow.add_edge("rephrase", "agent")
workflow.add_edge("generate", END)

app = workflow.compile()

graph_image = app.get_graph().draw_mermaid_png()
with open("src/api/v1/agent/Agent_workflow.png", "wb") as f:
    f.write(graph_image)


# =========================
# RUN AGENT
# =========================
def run_agent(query: str) -> dict:

    state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "iteration": 0,
        "validate": False,
        "tool_used": ""
    }

    final_state = app.invoke(state)

    if IRRELEVANT_MESSAGE.lower() in final_state["answer"].lower():
        return {
            "query": query,
            "answer": final_state["answer"],
            "retrieved_results": []
        }

    if final_state["tool_used"] == "nl2sql":
        return {
            "query": query,
            "answer": final_state["answer"],
            "sql_query": final_state.get("generated_sql"),
            "sql_result": final_state.get("sql_result"),
            "retrieved_results": []
        }

    top_chunks = final_state.get("reranked_docs", [])[:6]

    return {
        "query": query,
        "answer": final_state["answer"],
        "retrieved_results": [
            {
                "chunk_id": i + 1,
                "content": doc.page_content,
                **doc.metadata
            }
            for i, doc in enumerate(top_chunks)
        ]
    }