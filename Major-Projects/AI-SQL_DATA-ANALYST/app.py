# create environment for windows
# python -m venv myenv
# activate environment
# myenv\Scripts\activate
# pip install streamlit pandas matplotlib langchain 
# langchain-community langchain-groq python-dotenv

import streamlit as st
import time
import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI SQL Data Analyst",
    page_icon="📊",
    layout="wide"
)

# ---------------- SESSION ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

if "conn" not in st.session_state:
    st.session_state.conn = None

# ---------------- CSS ---------------- #
st.markdown("""
<style>

h1{
    font-weight:700;
}

.user-message{
    display:flex;
    justify-content:flex-end;
    margin-top:15px;
}

.user-bubble{
    background:#2563eb;
    color:white;
    padding:16px 22px;
    border-radius:20px 20px 4px 20px;
    max-width:45%;
    font-size:17px;
}

.ai-message{
    display:flex;
    justify-content:flex-start;
    margin-top:15px;
}

.ai-bubble{
    background:#262730;
    color:white;
    padding:15px;
    border-radius:18px 18px 18px 4px;
    width:70%;
}

.sql-box{
    background:#111827;
    color:#00ff88;
    padding:12px;
    border-radius:10px;
    margin-top:10px;
    font-family:monospace;
}

div[data-testid="stChatInput"]{
    position:fixed;
    bottom:20px;
    left:18%;
    width:68%;
}


</style>
""", unsafe_allow_html=True)

def generate_chart(df):

    if len(df.columns) < 2:
        return None

    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    if pd.api.types.is_numeric_dtype(y):

        fig, ax = plt.subplots(figsize=(8,4))

        ax.bar(x.astype(str), y)

        plt.xticks(rotation=45)

        return fig

    return None

# ---------------- TITLE ---------------- #

st.markdown(
    """
    <h1 style='text-align:center;'>
        📊 AI SQL Data Analyst Agent
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Ask questions about your CSV using AI</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("📁 Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.session_state.df = df

        conn = sqlite3.connect("data.db")

        df.to_sql(
            "data_table",
            conn,
            if_exists="replace",
            index=False
        )

        st.session_state.conn = conn

        st.success("Dataset Loaded Successfully ✅")

    st.divider()

    st.header("📊 Dataset Information")

    if st.session_state.df is not None:

        df = st.session_state.df

        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        st.metric("Missing Values", df.isna().sum().sum())

        st.metric(
            "Queries",
            len(
                [
                    m for m in st.session_state.messages
                    if m["role"] == "user"
                ]
            )
        )

        with st.expander("📄 Preview Dataset"):
            st.dataframe(df.head(), use_container_width=True)

        with st.expander("📋 Column Names"):
            st.write(df.columns.tolist())

        if st.button(
            "🗑 Clear Chat",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.rerun()

    else:

        st.info("Upload a CSV file.")


# ---------------- DISPLAY CHAT ---------------- #

for msg in st.session_state.messages:

    if msg["role"]=="user":

        st.markdown(f"""
        <div class="user-message">
            <div class="user-bubble">
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("### 🤖 AI SQL Analyst")

        with st.expander("🧠 View Generated SQL"):

            st.code(msg["sql"], language="sql")

        st.markdown("#### 📊 Result")

        if isinstance(msg["result"], pd.DataFrame):

            st.dataframe(
                msg["result"],
                use_container_width=True
            )

            st.caption(
                f"⚡ Executed in {msg['execution_time']} seconds"
            )

            csv = msg["result"].to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Result CSV",
                data=csv,
                file_name="query_result.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"download_{hash(msg['sql'])}"
            )

            st.markdown("#### 💡 AI Insight")
            st.info(msg["insight"])

        else:
            st.write(msg["result"])

    if msg.get("chart") is not None:

        st.markdown("#### 📈 Visualization")

        st.pyplot(msg["chart"])
        
if st.session_state.df is None:

    st.warning("📂 Upload a CSV file to begin.")

else:

    question = st.chat_input("Ask anything about your data...")

    if question:

        with st.spinner("🧠 AI is analyzing your data..."):

            prompt = f"""
You are an expert SQLite developer.

Database:
data_table

Columns:
{', '.join(st.session_state.df.columns)}

Generate ONLY a valid SQLite SELECT query.

Rules:

1. Output only SQL.
2. Never explain.
3. Never use markdown.
4. Never use ```sql.
5. Never use SQL:
6. Never invent column names.
7. Never use unsupported SQL syntax.
8. Always use only the provided columns.
9. Always generate executable SQLite queries.

User Question:
{question}
"""
            
            st.session_state.messages.append({
    "role": "user",
    "content": question
})
            
            response = llm.invoke(prompt)

            query = response.content.strip()

            query = query.replace("```sql", "")
            query = query.replace("```", "")
            query = query.replace("SQL", "")
            query = query.replace("sql", "")
            query = query.strip()

            try:

                start = time.time()
                result_df = pd.read_sql_query(
                    query,
                    st.session_state.conn
                )

                end = time.time()
                execution_time = round(end - start, 4)

                insight_prompt = f"""
                    You are a data analyst.

                    The user asked:

                    {question}

                    The SQL executed:

                    {query}

                    Here is the result:

                    {result_df.head(10).to_string(index=False)}

                    Write a short explanation in 2–4 sentences.

                    Don't mention SQL.

                    Explain the result in simple English.
                    """

                insight = llm.invoke(insight_prompt).content

            except Exception as e:

                st.error(e)

                st.stop()

            # Generate chart
            fig = generate_chart(result_df)

            st.session_state.messages.append({
                "role":"assistant",
                "sql":query,
                "result":result_df,
                "chart":fig,
                "insight":insight,
                "execution_time":execution_time
            })
        
            st.rerun()
            
# ---------------- CHAT HISTORY ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []


