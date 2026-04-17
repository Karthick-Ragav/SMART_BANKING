import streamlit as st
import requests
import pandas as pd
import os

# -------------------------
# CONFIG
# -------------------------
BASE_URL = "http://localhost:8000"
QUERY_API = f"{BASE_URL}/api/v1/query"
UPLOAD_API = f"{BASE_URL}/api/v1/admin/upload"

ADMIN_PASSWORD = "admin123"

st.set_page_config(page_title="Smart Banking", layout="wide")

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "uploading" not in st.session_state:
    st.session_state.uploading = False

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("Controls")

    # Clear Chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Admin Section
    st.subheader("Admin Panel")

    if not st.session_state.admin_logged_in:
        password = st.text_input("Enter Password", type="password")

        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Logged in")
                st.rerun()
            else:
                st.error("Wrong password")

    else:
        st.success("Admin Logged In")

        uploaded_file = st.file_uploader(
            "Upload file for ingestion",
            type=["pdf", "txt"]
        )

        if uploaded_file:
            if st.button("Upload & Ingest", disabled=st.session_state.uploading):
                st.session_state.uploading = True

                with st.spinner("Uploading and processing..."):
                    try:
                        res = requests.post(
                            UPLOAD_API,
                            files={"file": uploaded_file}
                        )

                        if res.status_code == 200:
                            data = res.json()
                            st.success("File ingested successfully!")
                            st.info(f"Chunks created: {data.get('chunks_created', 0)}")
                        else:
                            st.error(f"Upload failed: {res.text}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

                st.session_state.uploading = False

# -------------------------
# MAIN CHAT UI
# -------------------------
st.title("NORTHSTAR BANK Chatbot")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
query = st.chat_input("Ask your question...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    QUERY_API,
                    json={"query": query}
                )

                if response.status_code == 200:
                    data = response.json()

                    answer = data.get("answer", "No answer found")
                    chunks = data.get("retrieved_results", [])
                    sql_query = data.get("sql_query")
                    sql_result = data.get("sql_result")

                    # -------------------------
                    # SHOW ANSWER
                    # -------------------------
                    st.markdown(answer)

                    # -------------------------
                    # SHOW SQL (if NL2SQL)
                    # -------------------------
                    if sql_query:
                        with st.expander("SQL Query"):
                            st.code(sql_query, language="sql")

                    if sql_result:
                        with st.expander("SQL Result"):
                            try:
                                # Try converting to table
                                df = pd.DataFrame(eval(sql_result))
                                st.dataframe(df)
                            except:
                                st.text(sql_result)

                    # -------------------------
                    # SHOW CHUNKS (if RAG)
                    # -------------------------
                    if chunks:
                        with st.expander("Retrieved & Reranked Chunks"):
                           st.json(chunks)
                    # Save response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")