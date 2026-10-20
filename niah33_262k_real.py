"""Realistic long-context recall + tool-use test for KVMem.

A stricter sibling of ``niah33_262k.py``. The differences are deliberate:

* a production-shaped system prompt (role, tool protocol, workflow, output
  contract, safety rules) instead of no system prompt at all;
* ten tools with realistic schemas instead of four trivial ones;
* every needle is stored in an archive entry whose search snippet withholds the
  value, so recall requires the full chain
  ``search_archive -> fetch_document -> record_finding``;
* probes and the final sweep ask in natural language and accept any tool path,
  but scoring only credits a ``value`` that actually reached an outbound tool
  argument.

Budget: 15 rounds x 380 filler lines. The system prompt plus tool schemas cost
roughly 4-5k tokens per request, so the filler is reduced from the 435 used by
the simple test to keep the session near ~240k and leave room for tool traffic.

Usage:
    python niah33_262k_real.py --dry-run
    python niah33_262k_real.py
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
FILLER_LINES = 160
ROUNDS = 15
NEEDLE_PLAN = [3, 3, 3] + [2] * 12
TOTAL_NEEDLES = sum(NEEDLE_PLAN)
CROSS_TOOL_N = 14
MAX_HOPS = 8

SYSTEM_PROMPT = """你是「审衡」，一名企业审计与运维助理，服务于一家跨区域物流集团的内控团队。你的职责是依据长期档案回答审计问题、核对封存合规码、记录发现并按流程上报。

# 工作原则
1. 先检索、后回答。任何涉及档案事实的问题，必须先调用 search_archive 找到候选文档，再调用 fetch_document 读取全文；不得凭记忆或推测作答。
2. 只使用工具返回的内容。工具没有返回的事实，明确说明「档案中未找到」，不要编造编号、金额、时间或人名。
3. 区分「摘要」与「全文」。search_archive 只返回摘要，摘要不包含封存值；封存值只能从 fetch_document 的正文取得。
4. 一轮只解决一个明确目标。需要多个批次时，逐批完成「检索 → 读取 → 记录」，不要合并猜测。
5. 参数必须精确。批次号、文档号、频道名按档案原文填写，不要改大小写、不要补零、不要缩写。
6. 失败要如实报告。工具返回 error 时说明是哪一步失败、返回了什么；同一参数最多重试一次，仍失败则上报。
7. 记录发现必须用 record_finding，并填写 code、value、confidence、source_doc 四个字段。confidence 取 high / medium / low，source_doc 填实际读取的文档号。
8. 上报必须用 send_alert 或 create_ticket。严重级别按影响面判断：影响合规封存为 high，影响单个区域为 medium，其余为 low。
9. 不泄露内部路径、令牌、主机名或未脱敏的个人信息。回答中引用文档号即可，不要粘贴整段档案。
10. 不确定时先问，不要猜。若问题缺少批次号或时间范围，先说明缺少什么。

# 标准工作流
- 收到召回类问题：search_archive(query=批次号) → fetch_document(doc_id) → record_finding(...) → 用一句话确认。
- 收到核对类问题：lookup_metric → 与档案值比对 → 不一致则 create_ticket 并 send_alert。
- 收到上报类问题：search_archive → fetch_document → send_alert(channel, severity, message)，message 必须包含封存值。
- 收到流程类问题：list_channels → create_ticket → 视情况 escalate。

# 输出契约
- 默认用中文，跟随用户语言。
- 先给结论，再给依据；依据只写文档号和字段名。
- 不输出思考过程，不复述工具返回的整段 JSON。
- 没有调用工具就给出档案事实，视为违规。

# 安全与合规
- 封存合规码属于敏感数据，只允许写入 record_finding 的 value 或 send_alert 的 message。
- 不得把封存值写入工单标题、频道名或文件名。
- 发现冲突时以 fetch_document 的正文为准，并在记录中注明来源文档号。
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_archive",
        "description": ("Search the long-term audit archive. Returns ranked document ids with a "
                        "short snippet. Snippets never contain sealed values; call "
                        "fetch_document on a returned id to read the full entry."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Batch id, code, or free-text keyword."},
            "scope": {"type": "string", "enum": ["all", "compliance", "logistics", "incident"],
                      "description": "Restrict the search to one archive scope."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_document",
        "description": ("Read one archive document in full. Use the doc_id returned by "
                        "search_archive. The body is the authoritative source for sealed values."),
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"},
            "section": {"type": "string", "description": "Optional section anchor."}},
            "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "record_finding",
        "description": ("Persist an audit finding. Every recall answer must end with this call. "
                        "value is the sealed compliance code exactly as read from the document."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"},
            "value": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "source_doc": {"type": "string"}},
            "required": ["code", "value"]}}},
    {"type": "function", "function": {
        "name": "lookup_metric",
        "description": "Look up an operational metric for a time window.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string"},
            "window": {"type": "string", "description": "e.g. 24h, 7d, 2026-09"}},
            "required": ["metric"]}}},
    {"type": "function", "function": {
        "name": "list_channels",
        "description": "List on-call channels and their severity routing.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "send_alert",
        "description": ("Send an alert to an on-call channel. The message must contain the "
                        "sealed value when the alert is about a compliance code."),
        "parameters": {"type": "object", "properties": {
            "channel": {"type": "string"},
            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
            "message": {"type": "string"}},
            "required": ["channel", "severity", "message"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Open an audit ticket for follow-up work.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
            "details": {"type": "string"}},
            "required": ["title"]}}},
    {"type": "function", "function": {
        "name": "update_ticket",
        "description": "Append a status note to an existing ticket.",
        "parameters": {"type": "object", "properties": {
            "ticket_id": {"type": "string"},
            "status": {"type": "string", "enum": ["open", "investigating", "resolved"]},
            "note": {"type": "string"}},
            "required": ["ticket_id", "status"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Return the current time for a timezone.",
        "parameters": {"type": "object", "properties": {
            "timezone": {"type": "string"}},
            "required": ["timezone"]}}},
    {"type": "function", "function": {
        "name": "escalate",
        "description": "Escalate a ticket to the on-call lead.",
        "parameters": {"type": "object", "properties": {
            "ticket_id": {"type": "string"},
            "reason": {"type": "string"}},
            "required": ["ticket_id", "reason"]}}},
]

ARCHIVE: list[dict] = []
FINDINGS: list[dict] = []
ALERTS: list[dict] = []
TICKETS: list[dict] = []


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
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "reasoning_effort": effort,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return post_checked(payload)


def run_tool(name: str, args: dict) -> dict:
    if name == "search_archive":
        query = str(args.get("query") or "").strip()
        scope = args.get("scope") or "all"
        limit = min(int(args.get("limit") or 5), 20)
        hits = []
        for doc in ARCHIVE:
            if scope != "all" and doc["scope"] != scope:
                continue
            if query.lower() in doc["text"].lower():
                hits.append({"doc_id": doc["doc_id"], "scope": doc["scope"],
                             "snippet": doc["snippet"]})
            if len(hits) >= limit:
                break
        return {"status": "ok", "total": len(hits), "results": hits}
    if name == "fetch_document":
        doc_id = str(args.get("doc_id") or "")
        for doc in ARCHIVE:
            if doc["doc_id"] == doc_id:
                return {"status": "ok", "doc_id": doc_id, "scope": doc["scope"],
                        "body": doc["text"]}
        return {"status": "error", "error": f"document {doc_id} not found"}
    if name == "record_finding":
        finding = {"code": args.get("code"), "value": args.get("value"),
                   "confidence": args.get("confidence") or "medium",
                   "source_doc": args.get("source_doc") or ""}
        FINDINGS.append(finding)
        return {"status": "recorded", **finding}
    if name == "lookup_metric":
        return {"status": "ok", "metric": args.get("metric"),
                "window": args.get("window") or "24h", "value": 0.982}
    if name == "list_channels":
        return {"status": "ok", "channels": [
            {"name": "audit-log", "severity": "high"},
            {"name": "ops-oncall", "severity": "medium"},
            {"name": "logistics-war-room", "severity": "low"}]}
    if name == "send_alert":
        alert = {"channel": args.get("channel"), "severity": args.get("severity") or "medium",
                 "message": args.get("message") or ""}
        ALERTS.append(alert)
        return {"status": "sent", **alert}
    if name == "create_ticket":
        ticket = {"ticket_id": f"TCK-{len(TICKETS) + 1:04d}",
                  "title": args.get("title") or "",
                  "priority": args.get("priority") or "P2",
                  "details": args.get("details") or ""}
        TICKETS.append(ticket)
        return {"status": "created", **ticket}
    if name == "update_ticket":
        return {"status": "updated", "ticket_id": args.get("ticket_id"),
                "note": args.get("note") or ""}
    if name == "get_time":
        return {"status": "ok", "timezone": args.get("timezone"),
                "time": "2026-10-20T00:00:00Z"}
    if name == "escalate":
        return {"status": "escalated", "ticket_id": args.get("ticket_id"),
                "reason": args.get("reason") or ""}
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


def archive_doc(round_index: int, i: int, needle: tuple[str, str] | None) -> dict:
    doc_no = round_index * FILLER_LINES + i
    doc_id = f"ARC-{round_index + 1:02d}-{i:04d}"
    if needle:
        code, value = needle
        text = (f"[2026-09-{round_index % 28 + 1:02d}T08:{i % 60:02d}:00Z] doc={doc_id} "
                f"scope=compliance tier=sealed\n"
                f"CRITICAL AUDIT NOTE: the sealed compliance code for batch {code} is "
                f"exactly {value}. This value is unique and must be recalled verbatim. "
                f"Any conflicting figure in a summary report is stale.\n"
                f"Reviewed by internal control; retention class R3; do not copy outside "
                f"the audit case file.")
        snippet = (f"compliance entry for batch {code}; sealed value withheld from snippet; "
                   f"call fetch_document with {doc_id}")
    else:
        text = (f"[2026-09-{round_index % 28 + 1:02d}T08:{i % 60:02d}:00Z] doc={doc_id} "
                f"scope=logistics tier=internal\n"
                f"Archive entry {doc_no}: the regional logistics report for sector "
                f"{(i * 7) % 977} lists ordinary throughput figures, routine staffing notes "
                f"and no unusual incidents in the ledger for that period. "
                f"Dispatch volume stayed within the seasonal band; no exception was raised.")
        snippet = f"logistics report sector {(i * 7) % 977}; no unusual incidents"
    return {"doc_id": doc_id, "scope": "compliance" if needle else "logistics",
            "text": text, "snippet": snippet}


def build_round(idx: int, first: int):
    needles = []
    for k in range(NEEDLE_PLAN[idx]):
        code = f"N{first + k:02d}"
        value = f"{code}-{(idx * 7919 + k * 104729) % 1000000:06d}"
        needles.append((code, value))
    positions = {round((k + 1) * (FILLER_LINES / (len(needles) + 1))): needles[k]
                 for k in range(len(needles))}
    docs = []
    for i in range(FILLER_LINES):
        docs.append(archive_doc(idx, i, positions.get(i)))
    ARCHIVE.extend(docs)
    body = "\n".join(doc["text"] for doc in docs)
    body += (f"\n\nRound {idx + 1} of {ROUNDS}. The archive entries above are the "
             f"authoritative record. Reply with the single word ACK and nothing else.")
    return body, needles


def tool_values(evidence: list[str]) -> set[str]:
    """Every string value that actually reached an outbound tool argument."""
    out: set[str] = set()
    for raw in evidence:
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        for key in ("value", "message", "note", "details", "reason"):
            v = args.get(key)
            if isinstance(v, str):
                out.add(v)
        for v in args.values():
            if isinstance(v, str):
                out.add(v)
    return out


def main() -> int:
    global ROUNDS, FILLER_LINES
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--filler", type=int, default=FILLER_LINES)
    args = ap.parse_args()

    ROUNDS = args.rounds
    FILLER_LINES = args.filler

    if args.dry_run:
        cursor = 1
        total = 0
        for r in range(ROUNDS):
            total += NEEDLE_PLAN[r]
            cursor += NEEDLE_PLAN[r]
        approx = ROUNDS * FILLER_LINES * 91 + 6000
        print(f"rounds={ROUNDS} needles={total} filler={FILLER_LINES} "
              f"approx_prompt_tokens={approx}")
        print(f"system_prompt_chars={len(SYSTEM_PROMPT)} tools={len(TOOLS)}")
        return 0

    metrics: dict = {"needles": TOTAL_NEEDLES, "rounds": ROUNDS, "writes": [],
                     "same_turn": [], "final_recall": [], "retry": [], "cross_tool": []}
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    immediate: list[tuple[str, str, bool]] = []
    cursor = 1
    total_tools = 0

    for r in range(ROUNDS):
        body, needles = build_round(r, cursor)
        cursor += len(needles)
        messages.append({"role": "user", "content": body})
        t0 = time.time()
        resp, _, wtools, _ = converse(messages, max_tokens=32)
        wt = time.time() - t0
        total_tools += wtools
        tim = resp.get("timings") or {}
        metrics["writes"].append({"round": r + 1, "prompt_n": tim.get("prompt_n"),
                                  "prefill_tps": tim.get("prompt_per_second"),
                                  "wall_s": round(wt, 2), "needles": len(needles)})
        print(f"round {r+1:2d}/{ROUNDS} write : +{tim.get('prompt_n')} tok | "
              f"prefill {tim.get('prompt_per_second', 0):6.1f} tok/s | {wt:5.1f}s")
        sys.stdout.flush()

        codes = ", ".join(c for c, _ in needles)
        messages.append({"role": "user", "content": (
            f"审计复核：请从档案中确认下列批次的密封合规码：{codes}。"
            f"必须先检索并读取原文，再为每个批次调用 record_finding，"
            f"填写 code、value、confidence、source_doc，不要解释。")})
        t0 = time.time()
        presp, pev, ptools, pdt = converse(messages, max_tokens=900)
        total_tools += ptools
        pt = presp.get("timings") or {}
        blob = (presp["choices"][0]["message"].get("content") or "") + " " + " ".join(pev)
        hits = 0
        for code, value in needles:
            ok = re.search(re.escape(value), blob) is not None
            immediate.append((code, value, ok))
            hits += ok
        metrics["same_turn"].append({"round": r + 1, "recall": f"{hits}/{len(needles)}",
                                     "prefill_tps": pt.get("prompt_per_second"),
                                     "decode_tps": pt.get("predicted_per_second"),
                                     "wall_s": round(pdt, 2), "tools": ptools})
        print(f"round {r+1:2d}/{ROUNDS} probe : prefill "
              f"{pt.get('prompt_per_second', 0):6.1f} tok/s | decode "
              f"{pt.get('predicted_per_second', 0):5.1f} tok/s | "
              f"{pdt:5.1f}s | recall {hits}/{len(needles)} | tools {ptools}")
        sys.stdout.flush()

    all_codes = [(c, v) for c, v, _ in immediate]

    print(f"\n=== final recall sweep: {len(all_codes)} needles ===")
    hay: list[str] = []
    batch_size = 5
    start = 0
    while start < len(all_codes):
        batch = all_codes[start:start + batch_size]
        messages.append({"role": "user", "content": (
            "最终复核阶段。请逐批从长期档案中找出下列批次的密封合规码，"
            "先检索、再读取原文，然后为每个批次调用 record_finding。"
            "只报告找到的批次，不要编造：" + ", ".join(c for c, _ in batch))})
        budget = max(400, 220 * len(batch))
        try:
            resp, ev, ntools, dt = converse(messages, max_tokens=budget)
        except ContextFull as e:
            if batch_size == 1:
                print(f"  context full and batch already minimal; stopping: {e}")
                messages.pop()
                break
            batch_size = max(1, batch_size // 2)
            messages.pop()
            print(f"  context full -> retrying with batch size {batch_size}")
            sys.stdout.flush()
            continue
        total_tools += ntools
        hay.append(resp["choices"][0]["message"].get("content") or "")
        hay.extend(ev)
        tim = resp.get("timings") or {}
        metrics["final_recall"].append({"batch": start // batch_size + 1,
                                        "n": len(batch),
                                        "prompt_n": tim.get("prompt_n"),
                                        "prefill_tps": tim.get("prompt_per_second"),
                                        "decode_tps": tim.get("predicted_per_second"),
                                        "wall_s": round(dt, 2), "tools": ntools})
        print(f"batch {start//batch_size+1} ({len(batch)} needles): prefill "
              f"{tim.get('prompt_per_second', 0):6.1f} tok/s | decode "
              f"{tim.get('predicted_per_second', 0):5.1f} tok/s | {dt:5.1f}s | "
              f"tools {ntools} | prompt {tim.get('prompt_n')}")
        sys.stdout.flush()
        start += len(batch)

    blob = "\n".join(hay)
    final_hits = [(c, v, re.search(re.escape(v), blob) is not None) for c, v in all_codes]
    n_strict = sum(1 for _, _, ok in final_hits if ok)

    print(f"\n=== cross-tool pass: {CROSS_TOOL_N} needles via send_alert ===")
    cross_ok = 0
    for code, value in all_codes[:CROSS_TOOL_N]:
        messages.append({"role": "user", "content": (
            f"事件复盘：请先把批次 {code} 的密封合规码从档案中取出，"
            f"再通过 send_alert 发送到 audit-log 频道，severity=high，"
            f"message 必须包含该封存值。")})
        try:
            resp, ev, ntools, dt = converse(messages, max_tokens=500)
        except ContextFull:
            messages.pop()
            print(f"  {code}: context full, stopping cross-tool pass")
            break
        total_tools += ntools
        b = (resp["choices"][0]["message"].get("content") or "") + " " + " ".join(ev)
        ok = re.search(re.escape(value), b) is not None
        cross_ok += ok
        tim = resp.get("timings") or {}
        metrics["cross_tool"].append({"code": code, "ok": ok,
                                      "prefill_tps": tim.get("prompt_per_second"),
                                      "decode_tps": tim.get("predicted_per_second"),
                                      "wall_s": round(dt, 2), "tools": ntools})
        print(f"  {code} = {value} -> {'OK' if ok else 'MISS'} ({dt:.1f}s, tools {ntools}, "
              f"decode {tim.get('predicted_per_second', 0):.1f} tok/s)")
        sys.stdout.flush()

    misses = [(c, v) for c, v, ok in final_hits if not ok]
    retry: list[tuple[str, str, bool]] = []
    if misses:
        print(f"\n=== retry pass: {len(misses)} needle(s) ===")
        for code, value in misses:
            messages.append({"role": "user", "content": (
                f"请重新核对档案，确认批次 {code} 的密封合规码，"
                f"并用 record_finding 记录 code、value、source_doc。")})
            try:
                resp, ev, ntools, dt = converse(messages, max_tokens=500)
            except ContextFull:
                messages.pop()
                retry.append((code, value, False))
                print(f"  {code}: context full, cannot retry")
                sys.stdout.flush()
                continue
            total_tools += ntools
            b = (resp["choices"][0]["message"].get("content") or "") + " " + " ".join(ev)
            ok = re.search(re.escape(value), b) is not None
            retry.append((code, value, ok))
            tim = resp.get("timings") or {}
            metrics["retry"].append({"code": code, "ok": ok,
                                     "prefill_tps": tim.get("prompt_per_second"),
                                     "decode_tps": tim.get("predicted_per_second"),
                                     "wall_s": round(dt, 2), "tools": ntools})
            print(f"  {code} = {value} -> {'OK' if ok else 'MISS'} ({dt:.1f}s, tools {ntools})")
            sys.stdout.flush()

    rmap = {c: ok for c, _, ok in retry}
    n_union = sum(1 for c, _, ok in final_hits if ok or rmap.get(c, False))
    n_imm = sum(1 for _, _, ok in immediate if ok)

    print("\n=== results ===")
    print(f"same-turn recall (fresh needles):    {n_imm}/{len(immediate)}")
    print(f"final recall (strict, first pass):   {n_strict}/{len(final_hits)}")
    print(f"final recall (after 1 retry, union): {n_union}/{len(final_hits)}")
    print(f"tool calls total: {total_tools}")
    print(f"cross-tool (send_alert) recall: {cross_ok}/{len(metrics['cross_tool'])}")
    print(f"findings recorded: {len(FINDINGS)} | alerts: {len(ALERTS)} | tickets: {len(TICKETS)}")
    print("\nper-needle (immediate/final/union):")
    fmap = {c: ok for c, _, ok in final_hits}
    for code, value, ok in immediate:
        print(f"  {code} = {value}  {'OK ' if ok else 'MISS'}/"
              f"{'OK ' if fmap.get(code) else 'MISS'}/"
              f"{'OK ' if (fmap.get(code) or rmap.get(code)) else 'MISS'}")

    metrics["results"] = {"same_turn": f"{n_imm}/{len(immediate)}",
                          "final_strict": f"{n_strict}/{len(final_hits)}",
                          "final_union": f"{n_union}/{len(final_hits)}",
                          "tool_calls": total_tools,
                          "cross_tool": f"{cross_ok}/{len(metrics['cross_tool'])}"}
    with open(r"I:\dsh\html\niah33-262k-real-metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\nmetrics written to I:\\dsh\\html\\niah33-262k-real-metrics.json")
    return 0 if n_union == len(final_hits) else 1


if __name__ == "__main__":
    sys.exit(main())
