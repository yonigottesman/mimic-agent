import os
from functools import partial
from textwrap import dedent
from typing import Literal

import requests
import yaml
from agents import TinyAgent, Tool, ToolsContainer, agentic_steps
from anthropic import Anthropic
from duckduckgo_search import DDGS
from markdownify import markdownify as md
from pydantic import BaseModel, Field
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax


def display_assistant_substep(message, console: Console):
    for content in message["content"]:
        if content.get("type", "") == "text" or "text" in content:
            console.print(Panel(content["text"], title="Reasoning", border_style="green"))
        elif content["type"] == "tool_use":
            text = Markdown(f"Using tool: **{content['name']}** with input:")
            inputs = Syntax(yaml.dump(content["input"]), "yaml", theme="monokai", line_numbers=False)
            console.print(Panel(Group(text, inputs), title="Tool Use", border_style="green"))
        elif content["type"] == "tool_result":
            console.print(Panel(content["content"], title="Tool Result", border_style="green"))


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


def fetch_web_page(inputs: FetchWebPageCommand, claude_client: Anthropic):
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
        model="claude-3-5-haiku-20241022",
        max_tokens=8192,
        system="",
        messages=[{"role": "user", "content": learning_prompt}],
        temperature=0.0,
    )
    return response.content[0].text


def main():
    console = Console()
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    tools = [
        Tool(search_web),
        Tool(fetch_web_page, call_args={"claude_client": client}),
    ]
    tools = ToolsContainer(tools)
    agent = TinyAgent(
        claude_client=client,
        tools=tools,
        system_prompt="You are a helpful assistant.",
        callback=partial(display_assistant_substep, console=console),
        model="claude-3-7-sonnet-20250219",
    )

    while True:
        user_input = console.input("> ")
        with console.status("[bold green]Thinking..."):
            answer = agent.run(user_input)
        console.print(Panel(Markdown(answer), title="Final Response", border_style="green"))


if __name__ == "__main__":
    main()
