# src/api/v1/agent/tools/hybrid_tool.py
from langchain_core.documents import Document
from langchain_core.tools import tool
from .vector_search_tool import vector_tool
from .fts_tool import fts_tool

@tool
def hybrid_tool(query: str, k: int = 10) -> list[Document]:
    "tool"
    vector_docs = vector_tool.invoke(query, k=k) 
    fts_docs = fts_tool.invoke(query, k=k)

    rrf_scores = {}
    doc_map = {}

    def get_unique_key(doc: Document) -> str:
        return f"{doc.page_content}_{doc.metadata.get('source','')}_{doc.metadata.get('page','')}"

    for rank, doc in enumerate(vector_docs):
        key = get_unique_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(fts_docs):
        key = get_unique_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        doc_map[key] = doc

    ranked_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [doc_map[k] for k in ranked_keys[:k]]