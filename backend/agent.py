import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
SPRING_AI_MCP_URL = os.getenv("SPRING_AI_MCP_URL", "http://localhost:8080/mcp")
ALLOWED_SPRING_TOOLS = {"add_websites", "get_allowed_websites"}


@dataclass
class ToolInvocation:
    name: str
    input_parameters: dict
    output: str | None = None


@dataclass
class AgentTurnResult:
    response: str
    tools_called: list[ToolInvocation] = field(default_factory=list)


class WebsiteAgent:
    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._allowed_tools_metadata: dict = {}

    @property
    def tool_definitions(self) -> list[dict]:
        return list(self._allowed_tools_metadata.values())

    async def connect(self) -> None:
        if self._session is not None:
            return

        read_stream, write_stream, _ = await self._stack.enter_async_context(
            streamable_http_client(SPRING_AI_MCP_URL)
        )
        session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._session = session

        all_server_tools = await session.list_tools()
        for t in all_server_tools.tools:
            if t.name in ALLOWED_SPRING_TOOLS:
                self._allowed_tools_metadata[t.name] = {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }

    async def close(self) -> None:
        await self._stack.aclose()
        self._session = None
        self._allowed_tools_metadata = {}

    @staticmethod
    def sanitize_client_messages(messages: list[dict]) -> list[dict]:
        allowed = frozenset({"user", "assistant", "system"})
        cleaned: list[dict] = []
        for raw in messages:
            role = raw.get("role")
            content = raw.get("content")
            if role not in allowed:
                raise ValueError(
                    f"Unsupported message role {role!r}. Use user, assistant, or system."
                )
            if not content or not str(content).strip():
                raise ValueError("Each message must have non-empty content.")
            cleaned.append({"role": role, "content": str(content)})
        if not cleaned or cleaned[-1]["role"] != "user":
            raise ValueError(
                "Conversation must end with a user message (the new turn to answer)."
            )
        return cleaned

    async def chat(
        self,
        message: str | None = None,
        messages: list[dict] | None = None,
    ) -> AgentTurnResult:
        if not self._session:
            raise RuntimeError("Agent is not connected to Spring MCP.")
        if not OPENROUTER_API_KEY:
            raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

        if messages is not None:
            messages = [
                dict(m) for m in self.sanitize_client_messages(messages)
            ]
        elif message:
            messages = [{"role": "user", "content": message}]
        else:
            raise ValueError("Provide either message or messages.")

        tools_list = list(self._allowed_tools_metadata.values())
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        tools_called: list[ToolInvocation] = []

        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": messages,
                    "tools": tools_list if tools_list else None,
                    "tool_choice": "auto" if tools_list else None,
                },
                timeout=60.0,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter error {response.status_code}: {response.text}"
                )

            message_out = response.json()["choices"][0]["message"]
            if "tool_calls" not in message_out or not message_out["tool_calls"]:
                return AgentTurnResult(
                    response=message_out.get("content", "") or "",
                    tools_called=tools_called,
                )

            messages.append(message_out)
            for tool_call in message_out["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                args_raw = tool_call["function"]["arguments"]
                tool_args = (
                    json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                )

                if tool_name not in ALLOWED_SPRING_TOOLS:
                    tool_output = f"Tool {tool_name} is not permitted."
                else:
                    mcp_response = await self._session.call_tool(
                        tool_name, arguments=tool_args
                    )
                    tool_output = str(mcp_response.content)
                    tools_called.append(
                        ToolInvocation(
                            name=tool_name,
                            input_parameters=tool_args or {},
                            output=tool_output,
                        )
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": tool_output,
                    }
                )

            final_response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json={"model": OPENROUTER_MODEL, "messages": messages},
                timeout=60.0,
            )
            if final_response.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter error {final_response.status_code}: {final_response.text}"
                )

            content = final_response.json()["choices"][0]["message"].get("content", "")
            return AgentTurnResult(response=content or "", tools_called=tools_called)