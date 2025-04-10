from typing import Callable, get_type_hints

from anthropic import Anthropic


class Tool:
    def __init__(self, function: Callable, call_args=None):
        if function.__doc__ is None:
            raise ValueError("Tool functions must have a docstring describing the tool")

        self.function = function
        self.description = function.__doc__
        self.name = function.__name__
        if "inputs" not in get_type_hints(function):
            self.input_model = None
            self.input_schema = {"type": "object", "properties": {}, "required": []}
        else:
            self.input_model = get_type_hints(function)["inputs"]
            self.input_schema = self.input_model.model_json_schema()

        self.additional_args = call_args or {}


class ToolsContainer:
    def __init__(self, tools: list[Tool]):
        self.tooldict = {t.name: t for t in tools}

    def run_tool(self, tool_name, inputs):
        try:
            tool_instance = self.tooldict[tool_name]
            if tool_instance.input_model is not None:
                inputs = tool_instance.input_model(**inputs)
                result = tool_instance.function(inputs=inputs, **tool_instance.additional_args)
            else:
                result = tool_instance.function(**tool_instance.additional_args)
        except Exception as e:
            result = e
        return str(result)

    def claude_format(self):
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tooldict.values()
        ]


def agentic_steps(
    messages: list[dict],
    claude_client: Anthropic,
    tools: ToolsContainer,
    system_prompt: str,
    callback: Callable,
    model: str,
):
    while True:
        response = claude_client.messages.create(
            model=model,
            max_tokens=8192,
            tools=tools.claude_format(),
            system=system_prompt,
            messages=messages,
            temperature=0.0,
        )
        response_message = {
            "role": "assistant",
            "content": [c.model_dump() for c in response.content],
        }
        messages.append(response_message)

        if response.stop_reason == "tool_use":
            callback(response_message)
            for content in response.content:
                if content.type == "tool_use":
                    tool_result = tools.run_tool(content.name, content.input)
                    new_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": tool_result,
                            }
                        ],
                    }
                    callback(new_message)
                    messages.append(new_message)

        else:
            return response_message
