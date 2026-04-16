# src/api/v1/agent/tools/vector_tool.py
import os
import pathlib
import base64
from langchain_core.documents import Document
from langchain_core.tools import tool
from src.core.db import get_db_conn, _embeddings_model  # make sure you have these

# -------------------------
# Convert DB row to Document
# -------------------------
def row_to_document(row) -> Document:
    return Document(
        page_content=row["content"],
        metadata={
            "chunk_type": row.get("chunk_type"),
            "page": row.get("page_number"),
            "section": row.get("section"),
            "source": row.get("source_file"),
            "element_type": row.get("element_type"),
            "image_path": row.get("image_base64"),  # already converted to base64
            "similarity": float(row.get("similarity", 0)),
            # "created_date": row.get("created_date")
        }
    )

# -------------------------
# Vector Tool
# -------------------------
@tool
def vector_tool(query: str, k: int = 10, chunk_type: str | None = None) -> list[Document]:
    """Return top-k vector similarity results as Document objects."""
    query_vec = _embeddings_model.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata,
            1 - (embedding <=> %(vec)s::vector) AS similarity
        FROM multimodal_chunks
        WHERE 1=1 {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"vec": embedding_str, "chunk_type": chunk_type, "k": k})
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        # Convert image to base64 if exists
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(pathlib.Path(img_path).read_bytes()).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    # Convert all rows to Document objects
    return [row_to_document(r) for r in results]