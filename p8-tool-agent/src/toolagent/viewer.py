"""Render trace files as one self-contained HTML page (no server, no CDN): filter by run, task or outcome, then
expand an episode to see every model turn, tool call, observation, retry and guardrail intervention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolagent.utils import read_jsonl

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent traces</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#e3e2de;--card:#fff;--ok:#1f7a4d;--bad:#b3261e;
--llm:#eef3fb;--tool:#f3f0fa;--obs:#f4f8f1;--guard:#fdf1e7;--err:#fbeceb}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe7;--mut:#9d9c96;--line:#33322f;--card:#1f1f1d;
--ok:#5cc08a;--bad:#ef8a80;--llm:#1d2635;--tool:#26213a;--obs:#1f2a1d;--guard:#33261a;--err:#3a1f1d}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0 12px}
.bar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
select,input{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);
color:var(--fg)}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:6px 0}
summary{cursor:pointer;padding:8px 10px;display:flex;gap:10px;flex-wrap:wrap;align-items:baseline}
.tag{font-size:12px;color:var(--mut)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.ev{margin:6px 10px;padding:6px 8px;border-radius:6px;white-space:pre-wrap;word-break:break-word;
font:12.5px/1.4 ui-monospace,Consolas,monospace}
.llm{background:var(--llm)}.tool_call{background:var(--tool)}.observation{background:var(--obs)}
.guard{background:var(--guard)}.tool_error,.format_error,.invalid_call,.retry{background:var(--err)}
.user,.end{background:var(--bg);border:1px dashed var(--line)}
.k{font-weight:700;font-family:system-ui,sans-serif}
</style></head><body><main>
<h1>Agent traces</h1>
<div class="bar"><select id="run"></select><select id="out"><option value="">all outcomes</option>
<option value="fail">failures only</option><option value="ok">successes only</option></select>
<input id="q" placeholder="filter task id" size="18"><span id="n" class="tag"></span></div>
<div id="list"></div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const runs = [...new Set(D.map(e => e.run))];
const sel = document.getElementById('run');
sel.innerHTML = '<option value="">all runs</option>' + runs.map(r => `<option>${esc(r)}</option>`).join('');
function body(ev) {
  const skip = new Set(['i', 'step', 'type']);
  const rest = Object.entries(ev).filter(([k]) => !skip.has(k))
    .map(([k, v]) => k === 'content' ? esc(v) : `${esc(k)}: ${esc(typeof v === 'object' ? JSON.stringify(v) : v)}`);
  return `<div class="ev ${esc(ev.type)}"><span class="k">#${ev.step} ${esc(ev.type)}</span>\\n${rest.join('\\n')}</div>`;
}
function render() {
  const r = sel.value, o = document.getElementById('out').value, q = document.getElementById('q').value.trim();
  const shown = D.filter(e => (!r || e.run === r) && (!q || e.task_id.includes(q)) &&
    (!o || (o === 'ok') === e.success));
  document.getElementById('n').textContent = `${shown.length} of ${D.length} episodes`;
  document.getElementById('list').innerHTML = shown.map(e => `<details><summary>
    <span class="${e.success ? 'ok' : 'bad'}">${e.success ? 'pass' : 'FAIL'}</span>
    <b>${esc(e.task_id)}</b><span class="tag">${esc(e.run)} · seed ${e.seed} · ${esc(e.failure)} ·
    ${e.steps} steps · end=${esc(e.end_reason)}</span></summary>${e.trace.map(body).join('')}</details>`).join('');
}
['run', 'out', 'q'].forEach(id => document.getElementById(id).addEventListener('input', render));
render();
</script></body></html>
"""


def load_episodes(trace_dir: str | Path) -> list[dict[str, Any]]:
    eps = []
    for p in sorted(Path(trace_dir).glob("*.jsonl")):
        for r in read_jsonl(p):
            s = r["score"]
            eps.append({"run": f"{r['model']} / {r['agent']}", "seed": r["seed"], "task_id": r["task_id"],
                        "success": bool(s["success"]), "failure": s["failure"], "steps": s["steps"],
                        "end_reason": s["end_reason"], "trace": r["trace"]})
    return eps


def render_html(episodes: list[dict[str, Any]]) -> str:
    data = json.dumps(episodes, default=str).replace("</", "<\\/")  # cannot close the <script> element early
    return PAGE.replace("__DATA__", data)


def write_viewer(trace_dir: str | Path, out_html: str | Path) -> int:
    eps = load_episodes(trace_dir)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(render_html(eps), encoding="utf-8")
    return len(eps)


__all__ = ["load_episodes", "render_html", "write_viewer"]
