#!/usr/bin/env python3
"""Run DeepEval metrics against eval/golden.json using the live MCP + OpenRouter stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    GEval,
    ToolCorrectnessMetric,
)
from deepeval.models import OpenRouterModel  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall  # noqa: E402

from backend.agent import (  # noqa: E402
    ALLOWED_SPRING_TOOLS,
    ToolInvocation,
    WebsiteAgent,
)

METRIC_THRESHOLD = 0.7
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden.json"
# Free OpenRouter models with structured-output support (avoid bare `openrouter/free`).
DEFAULT_JUDGE_MODEL = "baidu/ernie-5.5-flash-preview:free"


def build_judge_model() -> OpenRouterModel:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (see mcp-python-bridge/.env).")
    judge_model = os.getenv("DEEPEVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    print(f"DeepEval judge model: {judge_model}")
    return OpenRouterModel(
        model=judge_model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


def build_metrics(model: OpenRouterModel, profile: str = "all"):
    if profile == "tool-only":
        return [
            ToolCorrectnessMetric(
                threshold=METRIC_THRESHOLD,
                model=model,
                include_reason=True,
            )
        ]

    metric_kwargs = dict(threshold=METRIC_THRESHOLD, model=model, async_mode=False)
    tool_calling_accuracy = GEval(
        name="Tool Calling Accuracy",
        criteria=(
            "Determine whether the agent selected and invoked the correct MCP tools "
            f"({', '.join(sorted(ALLOWED_SPRING_TOOLS))}) for the user request. "
            "Score 1.0 when tool usage matches the task; 0.0 when wrong or missing tools."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.TOOLS_CALLED,
            SingleTurnParams.EXPECTED_TOOLS,
        ],
        threshold=METRIC_THRESHOLD,
        model=model,
    )
    tool_correctness = ToolCorrectnessMetric(
        include_reason=True,
        **metric_kwargs,
    )
    answer_relevancy = AnswerRelevancyMetric(
        include_reason=True,
        **metric_kwargs,
    )
    return [tool_calling_accuracy, tool_correctness, answer_relevancy]


def load_goldens(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("goldens", payload)


def to_tool_calls(specs: list[dict]) -> list[ToolCall]:
    return [
        ToolCall(
            name=item["name"],
            input_parameters=item.get("input_parameters") or item.get("inputParameters"),
        )
        for item in specs
    ]


def normalize_conversation(items: list) -> list[dict]:
    normalized: list[dict] = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"role": "user", "content": item})
        else:
            normalized.append({"role": item["role"], "content": item["content"]})
    return normalized


def format_conversation_input(history: list[dict]) -> str:
    lines = []
    for msg in history:
        if msg["role"] in ("user", "assistant"):
            lines.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(lines)


def expected_tools_for_golden(golden: dict) -> list[ToolCall]:
    if golden.get("expected_tools_by_turn"):
        flat: list[dict] = []
        for turn in golden["expected_tools_by_turn"]:
            flat.extend(turn)
        return to_tool_calls(flat)
    return to_tool_calls(golden.get("expected_tools") or [])


async def run_single_turn(agent: WebsiteAgent, golden: dict) -> tuple[str, str, list[ToolInvocation]]:
    user_input = golden["input"]
    result = await agent.chat(message=user_input)
    return user_input, result.response, list(result.tools_called)


async def run_multi_turn(
    agent: WebsiteAgent, golden: dict
) -> tuple[str, str, list[ToolInvocation]]:
    raw = golden.get("messages") or golden.get("conversation")
    if not raw:
        raise ValueError("Multi-turn golden requires messages or conversation.")

    history: list[dict] = []
    all_tools: list[ToolInvocation] = []
    last_response = ""

    for item in normalize_conversation(raw):
        if item["role"] == "assistant":
            history.append(item)
            continue

        history.append(item)
        turn = await agent.chat(messages=history.copy())
        last_response = turn.response
        history.append({"role": "assistant", "content": turn.response})
        all_tools.extend(turn.tools_called)
        print(
            f"    turn user={item['content'][:50]!r}… "
            f"tools={[t.name for t in turn.tools_called]}"
        )

    return format_conversation_input(history), last_response, all_tools


async def collect_test_cases(agent: WebsiteAgent, goldens: list[dict]) -> list[LLMTestCase]:
    cases: list[LLMTestCase] = []
    for golden in goldens:
        label = golden.get("name") or golden.get("input") or "conversation"
        print(f"\n→ Running golden: {label}")

        if "messages" in golden or "conversation" in golden:
            user_input, actual_output, invocations = await run_multi_turn(agent, golden)
        else:
            user_input, actual_output, invocations = await run_single_turn(agent, golden)

        tools_called = [
            ToolCall(
                name=t.name,
                input_parameters=t.input_parameters,
                output=t.output,
            )
            for t in invocations
        ]
        expected_tools = expected_tools_for_golden(golden)

        cases.append(
            LLMTestCase(
                input=user_input,
                actual_output=actual_output,
                expected_output=golden.get("expected_output"),
                tools_called=tools_called,
                expected_tools=expected_tools,
            )
        )
        print(
            f"  tools_called={[t.name for t in tools_called]} "
            f"expected={[t.name for t in expected_tools]}"
        )
    return cases


async def run_agent_goldens(
    golden_path: Path, limit: int | None = None, offset: int = 0
) -> list[LLMTestCase]:
    goldens = load_goldens(golden_path)
    if offset:
        goldens = goldens[offset:]
    if limit is not None:
        goldens = goldens[:limit]
    agent = WebsiteAgent()
    await agent.connect()
    try:
        return await collect_test_cases(agent, goldens)
    finally:
        await agent.close()


def open_results_ui() -> None:
    from deepeval.confident.api import is_confident

    if is_confident():
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "deepeval", "view"],
            cwd=ROOT,
            check=False,
        )
        return

    from eval.local_report import open_local_report

    path = open_local_report()
    print(
        "\nConfident AI signup blocked? Opened a local HTML report instead:\n"
        f"  {path}\n"
        "DeepEval also keeps JSON at .deepeval/.latest_run_full.json\n"
        "To try Confident later: support@confident-ai.com or API key after signup.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepEval golden-set runner")
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
        help=f"Path to golden JSON (default: {DEFAULT_GOLDEN})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N golden cases (e.g. --limit 1 for a smoke test).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N golden cases.",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default="website-mcp-eval",
        help="Run name shown in Confident AI.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open results in the browser (Confident if logged in, else local HTML report).",
    )
    parser.add_argument(
        "--metrics",
        choices=("all", "tool-only"),
        default="all",
        help="tool-only = ToolCorrectness only (fast, no judge LLM; good for free-tier smoke + Confident UI).",
    )
    args = parser.parse_args()
    if not args.golden.is_file():
        raise SystemExit(f"Golden file not found: {args.golden}")

    test_cases = asyncio.run(
        run_agent_goldens(args.golden, limit=args.limit, offset=args.offset)
    )
    model = build_judge_model()
    if args.metrics == "tool-only":
        print("Metrics profile: tool-only (ToolCorrectness; deterministic tool match)")
    metrics = build_metrics(model, profile=args.metrics)

    print(f"\nEvaluating {len(test_cases)} cases (threshold={METRIC_THRESHOLD})…\n")
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        identifier=args.identifier,
        async_config=AsyncConfig(max_concurrent=1, throttle_value=0),
    )
    if args.view:
        open_results_ui()


if __name__ == "__main__":
    main()