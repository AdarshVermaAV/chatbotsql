from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import mysql.connector
import os
import re
from datetime import datetime

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# If you have static files like CSS or JS, uncomment this
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Store chat history in memory (will be lost when server restarts)
chat_history = []

class TextInput(BaseModel):
    input_text: str

def process_text_input(text_input: TextInput) -> str:
    try:
        user_input = text_input.input_text.strip()

        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=512,
            request_timeout=10
        )

        classification_prompt = PromptTemplate.from_template("""
        You are a classification assistant for a chatbot.

        Classify the user's input into ONLY ONE of the following:
        - greet → greetings like "hi", "hello", etc.
        - explain → asking about your role or help
        - query → SQL-like questions about the 'chatbot' MySQL database
        - general → all other general questions (e.g. facts, jokes, science)

        Just respond with one word: greet, explain, query, or general.

        User Input: {question}
        Category:
        """)
        classify_chain = classification_prompt | llm
        decision = classify_chain.invoke({"question": user_input}).content.lower().strip()

        if decision == "greet":
            return "Hello! I'm your smart assistant — here to help with both database queries and general questions."

        elif decision == "explain":
            return (
                "I'm your smart AI assistant 🤖! Here's how I can help:\n"
                "- 💽 Run SQL queries and explore your database with ease\n"
                "- 🌐 Answer general questions — from fun facts to science, AI, and more\n"
                "- 🗣️ Have natural conversations about topics you're curious about\n"
                "\nJust ask me anything — I've got you covered! 🚀"
            )

        elif decision == "general":
            general_prompt = PromptTemplate.from_template("""
            You're a smart and friendly AI assistant 🤖.
            Respond to the user's general question in a helpful, natural, and conversational tone.
            Use emojis when appropriate to express emotions like curiosity, excitement, or empathy.
            Keep your answer clear, concise, and engaging.

            Question: {question}
            Answer:
            """)
            general_chain = general_prompt | llm
            general_response = general_chain.invoke({"question": user_input})
            return general_response.content if hasattr(general_response, "content") else str(general_response)

        mysql_uri = "mysql+pymysql://root:@127.0.0.1:3306/chatbot"
        db = SQLDatabase.from_uri(mysql_uri, include_tables=None, sample_rows_in_table_info=2)

        sql_prompt = """
        You are an expert MySQL query writer. Based on the schema below, write a MySQL query only that would answer the user's question.
        CONTEXT:
        - You are working exclusively with the 'chatbot' MySQL database.
        - Always use 'chatbot' as the table schema when querying information_schema.
        - Never assume table names or contents — only use what's in the schema.

        IMPORTANT:
        - Assume the database schema is named `chatbot`. Use this as the database name always.
        - If the user asks how many tables are in the database, use:
            SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'chatbot';                       
        - NEVER return labels like "SQL Query:" or "Here is the query:"        
        - If the user asks for table names, query table_name from information_schema.tables where table_schema = 'chatbot'.
        - If the user asks about columns, use information_schema.columns.
        - NEVER return markdown formatting like ```sql or comments or explanations.
        - Output only a valid SQL query that can be directly executed.
        - ONLY output a SQL query that can be executed directly on the `chatbot` database.

        {schema}

        Question: {question}
        SQL Query:
        """
        prompt = ChatPromptTemplate.from_template(sql_prompt)
        sql_chain = create_sql_query_chain(llm, db)
        response = sql_chain.invoke({"question": user_input})

        raw_sql = response if isinstance(response, str) else response.get("query", "")
        sql_query = re.sub(r"```sql|```|SQLQuery:|^sql\s*:\s*", "", raw_sql, flags=re.IGNORECASE).strip()

        if not sql_query or not sql_query.lower().startswith(("select", "insert", "update", "delete", "show", "describe")):
            return "Sorry, I couldn't generate a valid SQL query. Please ask a specific database-related question."

        conn = mysql.connector.connect(
            host="127.0.0.1",
            port=3306,
            user="root",
            password="",
            database="chatbot"
        )

        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()

        def sanitize_data(rows):
            sanitized = []
            for row in rows:
                clean_row = []
                for item in row:
                    if isinstance(item, str) and ("$2y$" in item or "@" in item):
                        clean_row.append("[REDACTED]")
                    else:
                        clean_row.append(item)
                sanitized.append(tuple(clean_row))
            return sanitized

        safe_rows = sanitize_data(rows)

        result_template = """
        Provide a clean, natural answer in paragraph format from this result:
        Result: {context}
        Question: {question}
        """
        answer_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=result_template
        )
        llm_chain = answer_prompt | llm
        final_answer_obj = llm_chain.invoke({"context": str(safe_rows), "question": user_input})
        final_answer = final_answer_obj.content if hasattr(final_answer_obj, "content") else str(final_answer_obj)

        return final_answer

    except mysql.connector.Error as db_err:
        fallback_prompt = PromptTemplate.from_template("""
        You're a helpful AI assistant.
        A database error occurred while trying to run a query for the following question.

        Kindly give a conversational, helpful fallback answer — estimate what the user likely meant and help them.

        Question: {question}
        Error: {error}
        Answer:
        """)
        fallback_chain = fallback_prompt | llm
        fallback_answer = fallback_chain.invoke({
            "question": user_input,
            "error": db_err.msg
        })
        return fallback_answer.content if hasattr(fallback_answer, "content") else str(fallback_answer)

    except Exception as e:
        return f"Unexpected error occurred: {e}"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    current_time = datetime.now().strftime("%I:%M %p")
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "previous_messages": chat_history,
        "current_time": current_time
    })

@app.post("/process/", response_class=HTMLResponse)
async def process_input(request: Request, input_text: str = Form(...)):
    text_input = TextInput(input_text=input_text)
    output_text = process_text_input(text_input)
    current_time = datetime.now().strftime("%I:%M %p")
    chat_history.append({
        "question": input_text, 
        "answer": output_text,
        "time": current_time
    })
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "previous_messages": chat_history,
        "current_time": current_time
    })

@app.post("/process-text/")
async def process_text_endpoint(text_input: TextInput):
    output_text = process_text_input(text_input)
    return {"output_text": output_text}

@app.get("/clear/", response_class=HTMLResponse)
async def clear_history():
    chat_history.clear()
    return RedirectResponse(url="/", status_code=303)

@app.post("/clear-history/")
async def clear_history():
    chat_history.clear()
    return {"status": "cleared"}
