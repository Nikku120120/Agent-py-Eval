"""Open a browser report from DeepEval's local cache (no Confident AI account)."""

from __future__ import annotations

import html
import json
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEST_RUN_JSON = ROOT / ".deepeval" / ".latest_run_full.json"
REPORT_HTML = ROOT / ".deepeval" / "latest_report.html"


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def build_report_html(run: dict) -> str:
    identifier = run.get("identifier", "eval")
    passed = run.get("testPassed", 0)
    failed = run.get("testFailed", 0)
    duration = run.get("runDuration", 0)
    cases = run.get("testCases") or []

    metric_rows = []
    for m in run.get("metricsScores") or []:
        scores = m.get("scores") or []
        avg = sum(scores) / len(scores) if scores else 0.0
        metric_rows.append(
            f"<tr><td>{_esc(m.get('metric'))}</td>"
            f"<td>{avg:.2f}</td>"
            f"<td>{m.get('passes', 0)}</td>"
            f"<td>{m.get('fails', 0)}</td></tr>"
        )

    case_blocks = []
    for i, tc in enumerate(cases):
        metrics_html = []
        for md in tc.get("metricsData") or []:
            ok = md.get("success")
            badge = "pass" if ok else "fail"
            metrics_html.append(
                f'<details class="metric"><summary>'
                f'<span class="badge {badge}">{_esc(md.get("name"))}</span> '
                f'score {_esc(md.get("score"))} (threshold {_esc(md.get("threshold"))})'
                f"</summary>"
                f'<pre class="reason">{_esc(md.get("reason"))}</pre>'
                f"</details>"
            )
        tools_called = ", ".join(
            t.get("name", "") for t in (tc.get("toolsCalled") or [])
        )
        expected = ", ".join(
            t.get("name", "") for t in (tc.get("expectedTools") or [])
        )
        case_blocks.append(
            f'<section class="case">'
            f'<h2>Case {i + 1}: { _esc(tc.get("name")) }</h2>'
            f'<p class="meta"><strong>Input:</strong> {_esc(tc.get("input"))}</p>'
            f'<p class="meta"><strong>Tools called:</strong> {_esc(tools_called)}</p>'
            f'<p class="meta"><strong>Expected tools:</strong> {_esc(expected)}</p>'
            f'<details><summary>Actual output</summary>'
            f'<pre>{_esc(tc.get("actualOutput"))}</pre></details>'
            f"{''.join(metrics_html)}"
            f"</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DeepEval — {_esc(identifier)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 52rem; line-height: 1.45; }}
    h1 {{ font-size: 1.35rem; }}
    .summary {{ background: #f4f4f5; padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #eee; }}
    .case {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
    pre {{ white-space: pre-wrap; font-size: 0.85rem; background: #fafafa; padding: 0.75rem; border-radius: 6px; }}
    .badge {{ font-weight: 600; }}
    .badge.pass {{ color: #0a7; }}
    .badge.fail {{ color: #c33; }}
    .metric {{ margin: 0.5rem 0; }}
    .reason {{ font-size: 0.85rem; }}
    .note {{ color: #555; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>DeepEval run: {_esc(identifier)}</h1>
  <p class="note">Local report (Confident AI not required). Source: .deepeval/.latest_run_full.json</p>
  <div class="summary">
    <p><strong>Passed:</strong> {passed} &nbsp; <strong>Failed:</strong> {failed}</p>
    <p><strong>Duration:</strong> {duration:.2f}s</p>
  </div>
  <h2>Aggregate metrics</h2>
  <table>
    <tr><th>Metric</th><th>Avg score</th><th>Passes</th><th>Fails</th></tr>
    {''.join(metric_rows)}
  </table>
  <h2>Test cases</h2>
  {''.join(case_blocks)}
</body>
</html>"""


def open_local_report() -> Path:
    if not LATEST_RUN_JSON.is_file():
        raise FileNotFoundError(
            f"No local run found at {LATEST_RUN_JSON}. Run eval/run_deepeval.py first."
        )
    run = json.loads(LATEST_RUN_JSON.read_text(encoding="utf-8"))
    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_report_html(run)
    REPORT_HTML.write_text(html_text, encoding="utf-8")
    webbrowser.open(REPORT_HTML.resolve().as_uri())
    return REPORT_HTML