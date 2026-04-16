# src/api/v1/services/query_service.py

from typing import List
from src.api.v1.agent.agent import run_agent
from src.api.v1.schemas.query_schema import QueryResponse, RetrievedResult


def query_documents(query: str) -> QueryResponse:
    """
    Calls the agent and returns the final answer.
    Supports both:
    - RAG (retrieved chunks)
    - NL2SQL (SQL query + result)
    """

    agent_result = run_agent(query)

    # -------------------------
    # Map retrieved chunks (RAG)
    # -------------------------
    retrieved_results: List[RetrievedResult] = []

    for chunk in agent_result.get("retrieved_results", []):
        retrieved_results.append(
            RetrievedResult(
                chunk_id=chunk.get("chunk_id", 0),
                content=chunk.get("content", ""),
                chunk_type=chunk.get("chunk_type"),
                page=chunk.get("page"),
                section=chunk.get("section"),
                source=chunk.get("source"),
                image_path=chunk.get("image_path"),
                similarity=chunk.get("similarity"),
                created_date=chunk.get("created_date"),
                updated_date=chunk.get("updated_date"),
            )
        )

    # -------------------------
    # Extract SQL fields (NL2SQL)
    # -------------------------
    sql_query = agent_result.get("sql_query")
    sql_result = agent_result.get("sql_result")

    # -------------------------
    # Final Response
    # -------------------------
    return QueryResponse(
        query=agent_result.get("query", query),
        answer=agent_result.get("answer", "No relevant data found."),
        retrieved_results=retrieved_results,
        sql_query=sql_query,
        sql_result=sql_result,
    )