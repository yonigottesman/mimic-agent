import os
from functools import partial

import streamlit as st
import yaml
from anthropic import Anthropic
from google.cloud import bigquery

from agents import Tool, ToolsContainer, agentic_steps
from tools import get_highlevel_tables_information, get_table_schema_and_description, query_db

claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
bigquery_client = bigquery.Client(project=os.getenv("GCLOUD_PROJECT_ID"))

st.title("💬🗄️ MIMIC-III Agent")
st.caption("Chat with MIMIC-III tables")

if "messages" not in st.session_state:
    st.session_state.messages = []


def reset():
    st.session_state.messages = []


st.button("Clear History", on_click=reset)


def display_assistant_substep(message, expanded=False):
    with st.status("", expanded=expanded) as status:
        for content in message["content"]:
            if content["type"] == "text":
                status.write(content["text"])
            elif content["type"] == "tool_use":
                status.write(f"Using tool: **{content['name']}** with input:")
                if "query" in content["input"]:
                    assert len(content["input"]) == 1
                    status.code(content["input"]["query"], language="sql")
                else:
                    status.code(yaml.dump(content["input"]))
            elif content["type"] == "tool_result":
                status.write("Tool Result")
                status.code(content["content"])
        status.update(label="", state="complete", expanded=expanded)


def is_final_agentic_step(message):
    return len(message["content"]) == 1 and message["content"][0]["type"] == "text"


def display_messages(messages):
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "user" and all(
            m["type"] != "tool_result" if isinstance(m, dict) else True for m in messages[i]["content"]
        ):
            with st.chat_message("user"):
                st.markdown(messages[i]["content"])
        elif messages[i]["role"] == "assistant":
            with st.chat_message("assistant"):
                # run until last message in agentic steps
                while i < len(messages) and not (is_final_agentic_step(messages[i])):
                    display_assistant_substep(messages[i])
                    i += 1
                if i < len(messages):
                    st.markdown(messages[i]["content"][0]["text"])
        else:
            assert False
        i += 1


display_messages(st.session_state.messages)


def get_tools():
    tools = [
        # Tool(find_relevant_tables),
        Tool(get_table_schema_and_description),
        Tool(query_db, call_args={"client": bigquery_client}),
    ]
    return ToolsContainer(tools)


if "tools" not in st.session_state:
    st.session_state.tools = get_tools()


system_prompt = f"""
* You are a helpful assistant that can answer questions and access the MIMIC-III database.
* The database name is `physionet-data.mimiciii_clinical`.
* To access note events table use `physionet-data.mimiciii_notes` database.
* in the database the names of the tables are in lower snake_case.
* Before answering the user's question, you should first find the relevant tables that can answer the question.
* Be sure to check the schema of tables you want to access.
* Answer the user's question based on the information provided in the tables.
* You answer should be concise and to the point, dont provide more information than the user asks for.
* Here is some high level information about the tables in the database:
{get_highlevel_tables_information()}
"""

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking", show_time=True):
            answer = agentic_steps(
                messages=st.session_state.messages,
                claude_client=claude_client,
                tools=st.session_state.tools,
                system_prompt=system_prompt,
                callback=partial(display_assistant_substep, expanded=True),
                model="claude-3-7-sonnet-20250219",
            )
            st.markdown(answer["content"][0]["text"])
