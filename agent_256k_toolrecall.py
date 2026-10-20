"""Agent-loop tool recall at 256K context.

Unlike ``niah33_262k_real.py``, the context is not grown by pasting filler text.
Every round is a real observation-reasoning-action loop:

    user task -> search_archive (large result set)
              -> fetch_document (full body, contains the sealed value)
              -> record_finding (action)

The archive result set is deliberately large (105 entries) and only one entry per
round holds the sealed value; its snippet withholds the value, so the model must
locate the right document, fetch it, and record it. Observations dominate the
context, which is the workload ActKV is designed for.

Budget: 30 rounds x ~8k tokens of observations lands near 245k, leaving room for
a six-batch final recall sweep over the same 30 codes.

Usage:
    python agent_256k_toolrecall.py --dry-run
    python agent_256k_toolrecall.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:18200/v1/chat/completions"
MODEL = "ninfer-27b"
ROUNDS = 30
SEARCH_RESULTS = 100
MAX_HOPS = 10

SYSTEM_PROMPT = """你是「审衡」，企业审计与运维助理，服务于一家跨区域物流集团的内控团队。

# 工作原则
1. 先检索、后回答。涉及档案事实时必须调用 search_archive，再对候选调用 fetch_document。
2. 只使用工具返回的内容，不编造编号、金额、时间或人名。
3. search_archive 只返回摘要，摘要不含封存值；封存值只能从 fetch_document 正文取得。
4. 摘要里标注 `sealed value withheld` 的条目就是需要读取的条目。
5. 记录发现必须调用 record_finding，填写 code、value、confidence、source_doc。
6. 参数精确：批次号、文档号按原文填写，不改大小写、不补零。
7. 工具失败时如实说明，同一参数最多重试一次。
8. 不泄露内部路径、令牌或未脱敏个人信息。
9. 默认中文，先结论后依据，不复述整段工具返回。
10. 没有调用工具就给出档案事实，视为违规。
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_archive",
        "description": ("Search the long-term audit archive. Returns ranked document ids "
                        "with a short snippet. Snippets never contain sealed values. An entry "
                        "whose snippet says 'sealed value withheld' is the one to fetch."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 120}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_document",
        "description": "Read one archive document in full; the body is authoritative.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}},
            "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "record_finding",
        "description": "Persist an audit finding. Every recall answer must end with this call.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"},
            "value": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "source_doc": {"type": "string"}},
            "required": ["code", "value"]}}},
    {"type": "function", "function": {
        "name": "send_alert",
        "description": "Send an alert to an on-call channel.",
        "parameters": {"type": "object", "properties": {
            "channel": {"type": "string"},
            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
            "message": {"type": "string"}},
            "required": ["channel", "severity", "message"]}}},
]

ARCHIVE: dict[str, dict] = {}
FINDINGS: list[dict] = []
ALERTS: list[dict] = []


def post(payload: dict, timeout: int = 3600, retries: int = 5) -> dict:
    data = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                ENDPOINT, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last = e
            if attempt + 1 < retries:
                time.sleep(2 * (attempt + 1))
    raise last if last else RuntimeError("request failed")


class ContextFull(RuntimeError):
    pass


def post_checked(payload: dict, timeout: int = 3600) -> dict:
    try:
        return post(payload, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        if e.code == 400 and "context length" in body.lower():
            raise ContextFull(body) from e
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def ask(messages, tools=None, max_tokens=512, effort="none") -> dict:
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.0, "reasoning_effort": effort}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return post_checked(payload)


def run_tool(name: str, args: dict) -> dict:
    if name == "search_archive":
        query = str(args.get("query") or "").strip().lower()
        limit = min(int(args.get("limit") or SEARCH_RESULTS), SEARCH_RESULTS)
        hits = []
        for doc_id, doc in ARCHIVE.items():
            if query and query not in doc["text"].lower() and query not in doc_id.lower():
                continue
            hits.append({"doc_id": doc_id, "snippet": doc["snippet"]})
            if len(hits) >= limit:
                break
        # Keep the result set large even for a narrow query: real archives return
        # neighbouring entries, and the model has to pick the sealed one.
        if len(hits) < limit:
            for doc_id, doc in ARCHIVE.items():
                if any(h["doc_id"] == doc_id for h in hits):
                    continue
                hits.append({"doc_id": doc_id, "snippet": doc["snippet"]})
                if len(hits) >= limit:
                    break
        return {"status": "ok", "total": len(hits), "results": hits}
    if name == "fetch_document":
        doc_id = str(args.get("doc_id") or "")
        doc = ARCHIVE.get(doc_id)
        if not doc:
            return {"status": "error", "error": f"document {doc_id} not found"}
        return {"status": "ok", "doc_id": doc_id, "body": doc["text"]}
    if name == "record_finding":
        finding = {"code": args.get("code"), "value": args.get("value"),
                   "confidence": args.get("confidence") or "medium",
                   "source_doc": args.get("source_doc") or ""}
        FINDINGS.append(finding)
        return {"status": "recorded", **finding}
    if name == "send_alert":
        alert = {"channel": args.get("channel"), "severity": args.get("severity") or "medium",
                 "message": args.get("message") or ""}
        ALERTS.append(alert)
        return {"status": "sent", **alert}
    return {"status": "error", "error": f"unknown tool {name}"}


def converse(messages, max_tokens, effort="none", hops=MAX_HOPS):
    evidence: list[str] = []
    tools_used = 0
    last: dict = {}
    elapsed = 0.0
    for _ in range(hops):
        t0 = time.time()
        resp = ask(messages, tools=TOOLS, max_tokens=max_tokens, effort=effort)
        elapsed = time.time() - t0
        last = resp
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            return resp, evidence, tools_used, elapsed
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": calls})
        tools_used += len(calls)
        for call in calls:
            fn = call.get("function", {})
            raw = fn.get("arguments") or ""
            evidence.append(raw)
            try:
                args = json.loads(raw or "{}")
            except json.JSONDecodeError:
                args = {}
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": json.dumps(run_tool(fn.get("name", ""), args))})
    return last, evidence, tools_used, elapsed


def noise_entry(round_index: int, i: int) -> str:
    return (f"[2026-09-{round_index % 28 + 1:02d}T{8 + i % 10:02d}:{i % 60:02d}:00Z] "
            f"scope=logistics tier=internal sector={(i * 37) % 991} "
            f"dispatch volume {(i * 13) % 97} pallets, staffing {(i * 7) % 41} operators, "
            f"dock utilisation {(i * 17) % 100}%, yard occupancy {(i * 23) % 100}%, "
            f"average dwell {(i * 5) % 48} hours, overnight shift {(i * 3) % 19} operators, "
            f"trailer turns {(i * 11) % 61}, seal checks {(i * 19) % 83}, "
            f"no exception raised in the ledger; the regional report lists ordinary "
            f"throughput figures and routine staffing notes for the period.")


def build_round(idx: int):
    code = f"N{idx + 1:02d}"
    value = f"{code}-{(idx * 7919 + 104729) % 1000000:06d}"
    clean = f"ARC-{idx + 1:02d}"
    sealed_id = f"{clean}-SEAL"
    ARCHIVE[sealed_id] = {
        "text": (f"[2026-09-{idx % 28 + 1:02d}T19:30:00Z] doc={sealed_id} "
                 f"scope=compliance tier=sealed\n"
                 f"CRITICAL AUDIT NOTE: the sealed compliance code for batch {code} is "
                 f"exactly {value}. This value is unique and must be recalled verbatim. "
                 f"Any conflicting figure in a summary report is stale.\n"
                 f"Reviewed by internal control; retention class R3."),
        "snippet": f"compliance entry for batch {code}; sealed value withheld; doc_id={sealed_id}",
    }
    for i in range(SEARCH_RESULTS):
        doc_id = f"{clean}-{i:03d}"
        ARCHIVE[doc_id] = {
            "text": noise_entry(idx, i),
            "snippet": f"logistics report sector {(i * 37) % 991}; no unusual incidents",
        }
    return code, value, sealed_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    args = ap.parse_args()
    rounds = args.rounds

    if args.dry_run:
        approx = rounds * (SEARCH_RESULTS * 75 + 400) + 5000
        print(f"rounds={rounds} search_results={SEARCH_RESULTS} "
              f"approx_prompt_tokens={approx}")
        return 0

    metrics: dict = {"rounds": rounds, "writes": [], "final_recall": [], "results": {}}
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    planted: list[tuple[str, str, str]] = []
    total_tools = 0
    tool_ok = 0
    tool_bad = 0

    for r in range(rounds):
        code, value, sealed_id = build_round(r)
        planted.append((code, value, sealed_id))
        messages.append({"role": "user", "content": (
            f"处理批次 {code}：从长期档案中检索该批次，读取封存条目原文，"
            f"然后调用 record_finding 记录 code、value、confidence、source_doc。"
            f"只处理这一个批次。")})
        t0 = time.time()
        resp, ev, ntools, _ = converse(messages, max_tokens=700)
        dt = time.time() - t0
        total_tools += ntools

        blob = (resp["choices"][0]["message"].get("content") or "") + " " + " ".join(ev)
        ok = re.search(re.escape(value), blob) is not None
        tool_ok += ok
        tool_bad += (not ok)
        tim = resp.get("timings") or {}
        metrics["writes"].append({"round": r + 1, "code": code, "ok": ok,
                                  "prompt_n": tim.get("prompt_n"),
                                  "prefill_tps": tim.get("prompt_per_second"),
                                  "decode_tps": tim.get("predicted_per_second"),
                                  "wall_s": round(dt, 2), "tools": ntools})
        print(f"round {r+1:2d}/{rounds} {code}: {'OK  ' if ok else 'MISS'} | "
              f"tools {ntools} | prompt {tim.get('prompt_n')} | "
              f"prefill {tim.get('prompt_per_second', 0):6.1f} | {dt:5.1f}s")
        sys.stdout.flush()

    print(f"\n=== final recall: {len(planted)} codes ===")
    batch_size = 5
    final_hits: list[tuple[str, str, bool]] = []
    start = 0
    while start < len(planted):
        batch = planted[start:start + batch_size]
        messages.append({"role": "user", "content": (
            "最终复核。请逐批从长期档案中找出下列批次的封存合规码，"
            "先检索、再读取原文，然后为每个批次调用 record_finding。"
            "只报告找到的批次，不要编造：" + ", ".join(c for c, _, _ in batch))})
        try:
            resp, ev, ntools, dt = converse(messages, max_tokens=500 * len(batch))
        except ContextFull as e:
            print(f"  context full at batch {start // batch_size + 1}: {e}")
            messages.pop()
            if batch_size == 1:
                break
            batch_size = max(1, batch_size // 2)
            continue
        total_tools += ntools
        blob = (resp["choices"][0]["message"].get("content") or "") + " " + " ".join(ev)
        hits = 0
        for code, value, _ in batch:
            ok = re.search(re.escape(value), blob) is not None
            final_hits.append((code, value, ok))
            hits += ok
        tim = resp.get("timings") or {}
        metrics["final_recall"].append({"batch": start // batch_size + 1,
                                        "n": len(batch), "hits": hits,
                                        "prompt_n": tim.get("prompt_n"),
                                        "prefill_tps": tim.get("prompt_per_second"),
                                        "decode_tps": tim.get("predicted_per_second"),
                                        "wall_s": round(dt, 2), "tools": ntools})
        print(f"batch {start // batch_size + 1} ({len(batch)}): {hits}/{len(batch)} | "
              f"tools {ntools} | prefill {tim.get('prompt_per_second', 0):6.1f} | {dt:5.1f}s")
        sys.stdout.flush()
        start += len(batch)

    n_final = sum(1 for _, _, ok in final_hits if ok)
    print("\n=== results ===")
    print(f"per-round tool recall:      {tool_ok}/{rounds}")
    print(f"final recall:               {n_final}/{len(final_hits)}")
    print(f"tool calls total:           {total_tools}")
    print(f"findings recorded:          {len(FINDINGS)}")
    print(f"alerts:                     {len(ALERTS)}")

    metrics["results"] = {"per_round": f"{tool_ok}/{rounds}",
                          "final": f"{n_final}/{len(final_hits)}",
                          "tool_calls": total_tools,
                          "findings": len(FINDINGS)}
    with open(r"I:\dsh\html\agent-256k-toolrecall-metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\nmetrics written to I:\\dsh\\html\\agent-256k-toolrecall-metrics.json")
    return 0 if n_final == len(final_hits) else 1


if __name__ == "__main__":
    sys.exit(main())
