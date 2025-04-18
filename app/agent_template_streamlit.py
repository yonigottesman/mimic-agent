import os
from functools import partial
from textwrap import dedent
from typing import Literal

import requests
import streamlit as st
import yaml
from agents import Tool, ToolsContainer, agentic_steps
from anthropic import Anthropic
from duckduckgo_search import DDGS
from markdownify import markdownify as md
from pydantic import BaseModel, Field

st.title("💬🗄️ Agent Template")


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


@st.cache_resource
def get_claude_client():
    claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return claude_client


claude_client = get_claude_client()


class SearchWebCommand(BaseModel):
    query: str = Field(..., description="The query to search the web for")
    search_type: Literal["text", "news"] = Field(..., description="The type of search to perform")
    max_results: int = Field(..., description="The maximum number of results to return")


def search_web(inputs: SearchWebCommand):
    """Search the web using DuckDuckGo."""
    if inputs.search_type == "text":
        return str(DDGS().text(inputs.query, max_results=7))
    elif inputs.search_type == "news":
        return str(DDGS().news(inputs.query, max_results=7))
    else:
        raise ValueError(f"Invalid search type: {inputs.search_type}")


class FetchWebPageCommand(BaseModel):
    url: str = Field(..., description="The URL to fetch the web page from")
    required_learnings: str = Field(..., description="The required learnings to extract from the web page")
    high_level_goal: str = Field(..., description="The high level goal of why you need this web page")


def fetch_web_page(inputs: FetchWebPageCommand):
    """Fetch the web page from the given URL."""
    page_markdown = md(requests.get(inputs.url).text)
    learning_prompt = dedent(
        f"""
        Given a high level goal of a user request about the web page, and the web page itself,
        return the expected learnings from this web page.
        Return any relevant information that can help the user to achieve the high level goal.
        Be concise, return only the learnings you are requested.

        The high level goal is:
        {inputs.high_level_goal}
        The required learnings are:
        {inputs.required_learnings}
        The web page is:
        {page_markdown}
        """
    )
    response = claude_client.messages.create(
        model="claude-3-5-haiku-20241022",  # "claude-3-7-sonnet-20250219",
        max_tokens=8192,
        system="",
        messages=[{"role": "user", "content": learning_prompt}],
        temperature=0.0,
    )
    return response.content[0].text


def get_tools():
    tools = [
        Tool(search_web),
        Tool(fetch_web_page),
    ]
    return ToolsContainer(tools)


if "tools" not in st.session_state:
    st.session_state.tools = get_tools()


if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking", show_time=True):
            answer = agentic_steps(
                messages=st.session_state.messages,
                claude_client=claude_client,
                tools=st.session_state.tools,
                system_prompt="You are a helpful assistant",
                callback=partial(display_assistant_substep, expanded=True),
                model="claude-3-7-sonnet-20250219",
                max_steps=float("inf"),
            )
            st.markdown(answer)
