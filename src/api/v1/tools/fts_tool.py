# src/api/v1/agent/tools/fts_tool.py
import psycopg
from psycopg.rows import dict_row
import os
from langchain_core.documents import Document
from langchain_core.tools import tool

_CONN = os.getenv("PG_DSN")  # "postgresql://user:pass@localhost:5432/dbname"

@tool
def fts_tool(query: str, k: int = 10) -> list[Document]:
    "tool"
    sql = """
        SELECT
            mc.id,
            mc.content,
            mc.chunk_type,
            mc.page_number,
            mc.section,
            mc.source_file,
            mc.image_path,
            ts_rank(
                to_tsvector('english', mc.content),
                websearch_to_tsquery('english', %(query)s)
            ) AS score
        FROM multimodal_chunks mc
        LEFT JOIN documents d ON mc.doc_id = d.id
        WHERE to_tsvector('english', mc.content) @@ websearch_to_tsquery('english', %(query)s)
        ORDER BY score DESC
        LIMIT %(k)s;
    """
    with psycopg.connect(_CONN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query, "k": k})
            rows = cur.fetchall()

    docs = []
    for row in rows:
        row = dict(row)
        row["similarity"] = float(row["score"])
        docs.append(
            Document(
                page_content=row["content"],
                metadata={
                    "id": row.get("id"),
                    "chunk_type": row.get("chunk_type"),
                    "page": row.get("page_number"),
                    "section": row.get("section"),
                    "source": row.get("source_file"),
                    "image_path": row.get("image_path"),
                    "similarity": row.get("similarity"),
                    # "created_date": row.get("created_date"),
                    # "updated_date": None
                }
            )
        )
    return docs