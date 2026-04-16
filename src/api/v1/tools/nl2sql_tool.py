# src/api/v1/agent/tools/nl2sql_tool.py

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from src.core.db import get_sql_database
from langchain_google_genai import ChatGoogleGenerativeAI


_get_llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
@tool
def nl2sql_tool(query: str) -> dict:
    """
    Convert natural language query to SQL, execute it, and return structured answer.
    """

    llm = _get_llm
    db = get_sql_database()

    # -------------------------
    # Step 1: Generate SQL
    # -------------------------
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
        ("system",
"""You are a SQL expert.

Given the database schema below, write a single valid SELECT query that answers the user's question.

STRICT RULES:
- Return ONLY raw SQL (no explanation, no markdown, no backticks)
- Use ONLY tables and columns present in the schema
- Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements
- Add LIMIT 50 ONLY when necessary (not for aggregations or full lists)

DATABASE TABLES:

1. accounts
(account_id, customer_name, account_type, branch_code, ifsc_code, mobile, email, kyc_status, created_at)

2. transactions
(txn_id, account_id, txn_date, txn_type, amount, balance_after, description, channel, merchant_name, category, created_at)

3. loan_accounts
(loan_id, account_id, loan_type, principal, outstanding, disbursed_date, emi_amount, next_emi_date, interest_rate, tenure_months, emi_paid, status, created_at)

4. fixed_deposits
(fd_id, account_id, principal, interest_rate, tenure_days, start_date, maturity_date, maturity_amount, interest_payout, status, created_at)

5. credit_cards
(card_id, account_id, card_variant, credit_limit, available_limit, outstanding_amt, due_date, min_due, status, issued_date, created_at)

6. card_transactions
(txn_id, card_id, txn_date, txn_type, amount, merchant_name, category, is_international, currency, created_at)

RELATIONSHIPS:
- transactions.account_id → accounts.account_id
- loan_accounts.account_id → accounts.account_id
- fixed_deposits.account_id → accounts.account_id
- credit_cards.account_id → accounts.account_id
- card_transactions.card_id → credit_cards.card_id

IMPORTANT:
- Use JOIN when combining customer + transactions + loans + cards
- Use SUM, COUNT, AVG for aggregations
- Use WHERE for filters (account_id, loan_type, txn_type, status, etc.)
- Use ORDER BY for sorting
- Use DATE filters for time-based queries
- Prefer correct and simple queries

Database schema:
{schema}
"""),
        ("human", "Question: {question}")
    ])

    sql_chain = sql_prompt | llm

    raw_sql = sql_chain.invoke({
        "schema": schema_info,
        "question": query
    })

    content = raw_sql.content

    # Handle Gemini structured output
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )

    generated_sql = content.strip().strip("```").strip()

    if generated_sql.lower().startswith("sql"):
        generated_sql = generated_sql[3:].strip()

    print(f"\n[NL2SQL] Generated SQL:\n{generated_sql}")

    # -------------------------
    # Step 2: Execute SQL
    # -------------------------
    try:
        sql_result = db.run(generated_sql)
    except Exception as e:
        sql_result = f"SQL execution error: {e}"

    print(f"[NL2SQL] Result (truncated): {str(sql_result)[:200]}")

    # -------------------------
    # Step 3: Generate Answer
    # -------------------------
    

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful data analyst.

Rules:
- Answer ONLY using SQL results
- Be concise
- Format numbers clearly
- Do not hallucinate missing data
- If no data, say "No data found"
"""
        ),
        (
            "human",
            "Question: {query}\n\nSQL Used:\n{sql}\n\nQuery Results:\n{result}"
        )
    ])

    chain = answer_prompt | llm

    result = chain.invoke({
        "query": query,
        "sql": generated_sql,
        "result": sql_result
    })

    answer_text = (
        result.content[0]["text"]
        if isinstance(result.content, list)
        else str(result.content)
)

    print("[NL2SQL] Answer generated")

    # -------------------------
    # Return clean output
    # -------------------------
    return {
        "answer": answer_text,
        "sql_query": generated_sql,
        "sql_result": str(sql_result)
    }