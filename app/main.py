import os
from functools import partial

import streamlit as st
import yaml
from agents import Tool, ToolsContainer, agentic_steps
from anthropic import Anthropic
from google.cloud import bigquery
from tools import get_highlevel_tables_information, get_table_schema_and_description, query_db

claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
bigquery_client = bigquery.Client(project=os.getenv("GCLOUD_PROJECT_ID"))


MAX_MESSAGED = 20

st.title("💬🗄️ MIMIC-III Agent")
st.caption("Chat with MIMIC-III tables")
if "messages" not in st.session_state:
    st.session_state.messages = []


def reset():
    st.session_state.messages = []


st.button("Clear History", on_click=reset)


def display_assistant_substep(message, expanded=False):
    with st.status("", expanded=expanded) as status:
        label = ""
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
                expanded = False
                label = "Tool Result"
        status.update(label=label, state="complete", expanded=expanded)


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
        Tool(get_table_schema_and_description, call_args={"claude_client": claude_client}),
        Tool(query_db, call_args={"client": bigquery_client}),
    ]
    return ToolsContainer(tools)


if "tools" not in st.session_state:
    st.session_state.tools = get_tools()


system_prompt = f"""
* You are a helpful assistant that accesses the MIMIC-III database to answer questions.
* Database locations:
  - Clinical data: `mimiciii_clinical` (without the `physionet-data` prefix)
  - Notes data: `mimiciii_notes` (without the `physionet-data` prefix)
* Process for answering:
  1. Identify relevant tables for the question
  2. Check table schemas before querying
  3. Extract information from appropriate tables
  4. Provide concise, direct answers that address EXACTLY what was asked
  5. If in doubt, ask for more information
* Available tables overview:
{get_highlevel_tables_information()}
"""


if len(st.session_state.messages) > MAX_MESSAGED:
    st.caption("Maximum number of messages reached. Please clear history and try again.")
else:
    if prompt := st.chat_input("Ask me anything about MIMIC-III"):
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
                    max_steps=MAX_MESSAGED - len(st.session_state.messages),
                )
                st.markdown(answer)
