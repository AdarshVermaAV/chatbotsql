from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import mysql.connector
import os
import re
import json
from datetime import datetime

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/templates", StaticFiles(directory="templates"), name="templates")

chat_history = []

class TextInput(BaseModel):
    input_text: str

def process_text_input(text_input: TextInput):
    try:
        user_input = text_input.input_text.strip()

        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=512,
            request_timeout=10
        )

        # 1. Classification
        classification_prompt = PromptTemplate.from_template("""
        You are a classification assistant for a chatbot.

        Classify the user's input into ONLY ONE of the following:
        - greet → greetings like "hi", "hello", etc.
        - explain → asking about your role or help
        - query → database-related query
        - general → all other general questions (e.g. facts, jokes, science)

        Just respond with one word: greet, explain, query, or general.

        User Input: {question}
        Category:
        """)
        classify_chain = classification_prompt | llm
        decision = classify_chain.invoke({"question": user_input}).content.lower().strip()

        if decision == "greet":
            return {
                "response": "Hello! I'm your smart assistant — here to help with both database queries and general questions.",
                "name": "greeting"
            }

        elif decision == "explain":
            return (
                "I'm your smart AI assistant 🤖! Here's how I can help:\n"
                "- Run SQL queries and explore your database with ease\n"
                "- Answer general questions — from fun facts to science, AI, and more\n"
                "- Have natural conversations about topics you're curious about\n"
                "\nJust ask me anything — I've got you covered!"
            )

        elif decision == "general":
            general_prompt = PromptTemplate.from_template("""
            You're a smart and friendly AI assistant.
            Respond to the user's general question in a helpful, natural, and conversational tone.
            Use emojis when appropriate to express emotions like curiosity, excitement, or empathy.
            Keep your answer clear, concise, and engaging.

            Question: {question}
            Answer:
            """)
            general_chain = general_prompt | llm
            general_response = general_chain.invoke({"question": user_input})
            return {
                "response": general_response.content if hasattr(general_response, "content") else str(general_response),
                "name": "general"
            }

        # 2. SQL Processing
        mysql_uri = "mysql+pymysql://root:@127.0.0.1:3306/chatbot"
        db = SQLDatabase.from_uri(mysql_uri, include_tables=['talents'], sample_rows_in_table_info=2)

        sql_prompt = """
        You are a MySQL expert SQL query writer assistant. Based on the database schema, write an accurate SQL query that directly answers the user's question.

        CONTEXT:
        - You are working with the `chatbot` MySQL database.
        - The database may have multiple tables; JOIN them where relevant based on foreign keys or logical relationships.
        - Output only a valid SQL query, no explanations or markdown.
        - NEVER include comments or say "Here's the query".

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
            return {"response": "Sorry, I couldn't generate a valid SQL query. Please ask a specific database-related question.", "name": "invalid-sql"}

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
        columns = [desc[0] for desc in cursor.description]

        if not rows:
            fallback_prompt = PromptTemplate.from_template("""
            You're a helpful AI assistant.
            A database query returned no results for the following question.

            Kindly provide a helpful and friendly fallback answer based on your knowledge.

            Question: {question}
            Answer:
            """)
            fallback_chain = fallback_prompt | llm
            fallback_answer = fallback_chain.invoke({"question": user_input})
            return {
                "response": fallback_answer.content if hasattr(fallback_answer, "content") else str(fallback_answer),
                "name": "no-match"
            }

        def sanitize_data(row_dict):
            return {
                key: "[REDACTED]" if isinstance(value, str) and any(s in value.lower() for s in ["password", "card", "cvv", "$2y$", "@"]) else value
                for key, value in row_dict.items()
            }

        results = [sanitize_data(dict(zip(columns, row))) for row in rows]

       


        primary_identifier = extract_identifier(results[0])

        result_template = """
        Based on the provided data, craft a well-structured and professional biography in rich narrative style.

        Include the following sections ONLY if data is available:
        - Full Name
        - Summary Introduction
        - Education Background
        - Career Achievements
        - Awards & Recognition
        - Current Role

        Write this like a professional biography, with each section forming a coherent paragraph. Keep tone informative yet engaging. Avoid bullet points or section headings.

        Data: {context}
        Question: {question}

        Final Answer:
        """
        answer_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=result_template
        )
        llm_chain = answer_prompt | llm
        final_answer_obj = llm_chain.invoke({"context": json.dumps(results, ensure_ascii=False), "question": user_input})
        final_answer = final_answer_obj.content if hasattr(final_answer_obj, "content") else str(final_answer_obj)

        return {
            "name": primary_identifier,
            "response": final_answer
        }

    except mysql.connector.Error as db_err:
        return {"response": f"Database error: {db_err.msg}", "name": "db-error"}
    except Exception as e:
        return {"response": f"Unexpected error occurred: {e}", "name": "exception"}

def extract_identifier(row: dict) -> str:
    first = row.get("first_name") or row.get("firstname") or row.get("name")
    last = row.get("last_name") or row.get("lastname") or row.get("surname")
    full = row.get("full_name") or row.get("person_name")

    if full and isinstance(full, str):
        return full.strip().lower().replace(" ", "-")

    if first and last and isinstance(first, str) and isinstance(last, str):
        return f"{first.strip().lower()}-{last.strip().lower()}"

    if first and isinstance(first, str):
        return first.strip().lower().replace(" ", "-")

    return "unknown-user"
    
@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")

@app.post("/process-text/")
async def process_text_endpoint(text_input: TextInput):
    output_data = process_text_input(text_input)
    current_time = datetime.now().strftime("%I:%M %p")

    chat_history.append({
        "question": text_input.input_text,
        "answer": output_data.get("response"),
        "time": current_time
    })

    return {
        "output": output_data,
        "time": current_time
    }

@app.get("/get-history/")
async def get_history():
    return {"chat_history": chat_history}

@app.post("/clear-history/")
async def clear_history():
    chat_history.clear()
    return {"status": "cleared"}

@app.get("/get-time/")
async def get_time():
    current_time = datetime.now().strftime("%I:%M %p")
    return {"time": current_time}
