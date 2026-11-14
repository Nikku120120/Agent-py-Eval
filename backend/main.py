from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .agent import OPENROUTER_API_KEY, WebsiteAgent

load_dotenv()

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = WebsiteAgent()
    try:
        await agent.connect()
        app_state["agent"] = agent
        print(
            f"Agent locked down to MCP tools: "
            f"{[t['function']['name'] for t in agent.tool_definitions]}"
        )
        yield
    except Exception as e:
        print(f"Failed to establish connection to Spring AI: {e}")
        app_state["agent"] = None
        yield
    finally:
        await agent.close()
        app_state["agent"] = None


app = FastAPI(lifespan=lifespan)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Send full conversation history, or a single message for backward compatibility."""

    messages: list[ChatMessage] | None = None
    message: str | None = None

    @model_validator(mode="after")
    def require_messages_or_message(self):
        if not self.messages and not self.message:
            raise ValueError("Either messages or message is required.")
        return self


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    agent: WebsiteAgent | None = app_state.get("agent")
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Local Spring Boot MCP application is not reachable.",
        )
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Missing OPENROUTER_API_KEY environment variable.",
        )

    try:
        if request.messages:
            history = [m.model_dump() for m in request.messages]
            result = await agent.chat(messages=history)
        else:
            result = await agent.chat(message=request.message)
        return {"response": result.response}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e