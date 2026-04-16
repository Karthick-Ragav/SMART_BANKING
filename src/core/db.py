import base64
import hashlib
import json
import os
import pathlib
from langchain_postgres import PGVector
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Connection setup
#
# The .env connection string uses SQLAlchemy's dialect prefix
# "postgresql+psycopg://" so that LangChain can parse it.
# psycopg.connect() expects the standard "postgresql://" URI, so we strip
# the dialect marker before passing it to psycopg.
# ---------------------------------------------------------------------------
_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING", "")
_PG_DSN = _PG_CONNECTION.replace("postgresql+psycopg://", "postgresql://")

# How many chunks to embed per API call.
# Google's embedding API accepts up to 100 texts per batch.
_EMBED_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Issue 8 fix: Module-level embeddings singleton — avoids re-instantiating a
# new HTTP client on every store_chunks() / similarity_search() call.
# ---------------------------------------------------------------------------
_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536,
)

# ---------------------------------------------------------------------------
# Issue 9 fix: Lazy connection pool — reuses existing TCP connections instead
# of opening a new one per request. Created on first use to avoid failing at
# import time when the DB is not yet available (e.g. during tests).
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Return the module-level connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def get_db_conn():
    """Return a pooled connection context manager.

    Usage:
        with get_db_conn() as conn:
            with conn.cursor() as cur: ...
    """
    return _get_pool().connection()


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

def upsert_document(filename: str, source_path: str) -> str:
    """Insert a document record and return its UUID.

    Uses ON CONFLICT so re-ingesting the same filename updates the path
    and returns the *existing* doc_id rather than creating a duplicate.
    This makes ingestion idempotent at the document level.
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, source_path)
                VALUES (%s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET source_path = EXCLUDED.source_path,
                        ingested_at  = now()
                RETURNING id
                """,
                (filename, source_path),
            )
            row = cur.fetchone()
        conn.commit()
    return str(row["id"])


# ---------------------------------------------------------------------------
# Chunk storage
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[dict], doc_id: str) -> int:
    """Embed each chunk individually and insert it into the multimodal_chunks table."""
    if not chunks:
        return 0

    _DEDICATED_COLUMNS = {
        "content_type", "element_type", "section",
        "page_number", "source_file", "position", "image_base64",
    }

    all_embeddings = []
    # ── Embed one chunk at a time ───────────────────────────────
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        if not content.strip():
            print(f"Skipping empty chunk {i}")
            all_embeddings.append(None)
            continue
        try:
            emb = _embeddings_model.embed_documents([content])  # list of 1
            all_embeddings.extend(emb)
            print(f"Embedded chunk {i}/{len(chunks)}")
        except Exception as e:
            print(f"Failed to embed chunk {i}: {e}")
            all_embeddings.append(None)

    rows_inserted = 0
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Delete old chunks for this document
            cur.execute("DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid", (doc_id,))
            
            for chunk, embedding in zip(chunks, all_embeddings):
                meta = chunk.get("metadata", {})

                img_b64 = meta.get("image_base64")
                image_path: str | None = None
                mime_type = "image/png" if img_b64 else None
                if img_b64:
                    image_bytes = base64.b64decode(img_b64)
                    img_dir = pathlib.Path("data/images")
                    img_dir.mkdir(parents=True, exist_ok=True)
                    img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                    img_file = img_dir / f"{doc_id}_{img_hash}.png"
                    img_file.write_bytes(image_bytes)
                    image_path = str(img_file)

                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]" if embedding else None
                clean_meta = {k: v for k, v in meta.items() if k not in _DEDICATED_COLUMNS}

                cur.execute(
                    """
                    INSERT INTO multimodal_chunks (
                        doc_id, chunk_type, element_type, content,
                        image_path, mime_type,
                        page_number, section, source_file,
                        position, embedding, metadata
                    ) VALUES (
                        %s::uuid, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s::jsonb, %s::vector, %s::jsonb
                    )
                    """,
                    (
                        doc_id,
                        chunk.get("content_type"),
                        meta.get("element_type"),
                        chunk.get("content"),
                        image_path,
                        mime_type,
                        meta.get("page_number"),
                        meta.get("section"),
                        meta.get("source_file"),
                        json.dumps(meta.get("position")) if meta.get("position") else None,
                        embedding_str,
                        json.dumps(clean_meta),
                    ),
                )
                rows_inserted += 1

        conn.commit()

    print(f"Total rows inserted: {rows_inserted}")
    return rows_inserted

# ---------------------------------------------------------------------------
# Chunk listing (for preview / debugging)
# ---------------------------------------------------------------------------

def get_all_chunks(chunk_type: str | None = None, limit: int = 200) -> list[dict]:
    """Return all stored chunks, optionally filtered by type.

    Args:
        chunk_type: Optional filter — 'text', 'table', or 'image'.
        limit:      Max rows to return (default 200, safety cap).

    Returns:
        List of dicts with keys: id, content, chunk_type, page_number,
        section, source_file, element_type, image_base64, mime_type,
        position, metadata.
    """
    type_clause = "WHERE chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            id, content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata
        FROM multimodal_chunks
        {type_clause}
        ORDER BY page_number ASC NULLS LAST, id ASC
        LIMIT %(limit)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"chunk_type": chunk_type, "limit": limit})
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    return results


from langchain_community.utilities import SQLDatabase
import os


def get_sql_database() -> SQLDatabase:
    """
    Returns SQLDatabase for NL2SQL queries (read-only).
    """

    db_url = os.getenv("AGENTIC_RAG_DB_URL")

    if not db_url:
        raise ValueError("AGENTIC_RAG_DB_URL is not set")

    return SQLDatabase.from_uri(
        db_url,
        include_tables=["accounts", "transactions", "loan_accounts"],
        sample_rows_in_table_info=2,
    )