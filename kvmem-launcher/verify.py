#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一致性自检：确保 web/index.html 与 launcher.py 的契约没有漂移。

只依赖标准库，可随时运行：
    python verify.py

检查项：
  1. index.html 里的 FIELDS 与 launcher.py 的 FIELD_ORDER 逐项一致
     （顺序 / 键名 / 标志 / 类型）—— 这是命令行预览正确性的前提；
  2. JS 里引用的每个 DOM id 都存在于同一文件的 HTML 中；
  3. JS 里使用的每个 CSS class 都已在样式表中定义；
  4. 结构完整性：<script>/</script> 配对，且无未替换占位符；
  5. 下拉菜单（CHOICES）的每个取值都必须在 ninfer 服务端源码中出现过 ——
     即"选项都是服务端真实支持的"。若源码目录不存在则跳过该检查。

退出码 0 = 全部通过；1 = 存在不一致。
"""

import ast
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
HTML = BASE / "web" / "index.html"
PY = BASE / "launcher.py"

# ninfer 服务端源码（用于校验下拉取值）
# 本启动器位于仓库根目录 kvmem-launcher/，ninfer 源码相对其上两级；仓库外使用时回退绝对路径。
_REPO_ROOT = Path(__file__).resolve().parent.parent
NINFER_SRC = _REPO_ROOT / "backends" / "ninfer" / "src"
if not NINFER_SRC.is_dir():
    NINFER_SRC = Path(r"I:\llama\kvmem-llama.cpp-multi-backend-framework\backends\ninfer\src")

errors = []

# ----------------------------------------------------------------- 读取
if not HTML.exists():
    print("找不到 %s" % HTML)
    sys.exit(1)
html = HTML.read_text(encoding="utf-8")
src = PY.read_text(encoding="utf-8")

scripts = list(re.finditer(r"<script>([\s\S]*?)</script>", html))
if not scripts:
    print("index.html 中找不到 <script> 块")
    sys.exit(1)
# 主逻辑脚本 = 含 FIELDS 定义的那个（其余是 <head> 里的主题引导小脚本）
main_m = next((s for s in scripts if "const FIELDS" in s.group(1)), scripts[-1])
js = main_m.group(1)
head = html[:main_m.start()]

# ------------------------------------------------- 1) FIELDS 与 FIELD_ORDER
tree = ast.parse(src)
assign = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                try:
                    assign[t.id] = [[x.value for x in el.elts] for el in node.value.elts]
                except Exception:
                    pass

field_order = assign.get("FIELD_ORDER")
advanced = assign.get("ADVANCED_ORDER", [])
all_order = (field_order or []) + advanced
if field_order is None:
    errors.append("launcher.py 中找不到 FIELD_ORDER")

fm = re.search(r"const FIELDS = \[(.*?)\n\];", js, re.S)
if not fm:
    errors.append("index.html 中找不到 FIELDS 数组")
else:
    try:
        js_fields = [list(x) for x in json.loads("[" + fm.group(1).strip().rstrip(",") + "]")]
    except Exception as e:
        js_fields = None
        errors.append("FIELDS 数组解析失败: %s" % e)
    if js_fields is not None and field_order is not None:
        if js_fields == all_order:
            print("[1] FIELDS 一致：%d 项（基础 %d + 进阶 %d）"
                  % (len(js_fields), len(field_order), len(advanced)))
        else:
            errors.append("FIELDS 不一致：HTML %d 项 vs launcher.py %d 项"
                          % (len(js_fields), len(all_order)))
            for i in range(max(len(js_fields), len(all_order))):
                a = js_fields[i] if i < len(js_fields) else None
                b = all_order[i] if i < len(all_order) else None
                if a != b:
                    errors.append("    #%d HTML=%r PY=%r" % (i, a, b))

# --------------------------------------------------------------- 2) DOM id
head_ids = set(re.findall(r'id="([^"]+)"', head))
used_ids = (set(re.findall(r'\$\("([A-Za-z][\w]*)"\)', js))
            | set(re.findall(r'getElementById\("([^"]+)"\)', js)))
missing_ids = sorted(used_ids - head_ids)
if missing_ids:
    errors.append("JS 引用但 HTML 中不存在的 id: %s" % missing_ids)
else:
    print("[2] DOM id 校验通过：%d 个引用全部存在" % len(used_ids))

# ------------------------------------------------------------- 3) CSS class
sm = re.search(r"<style>([\s\S]*?)</style>", head)
if not sm:
    errors.append("HTML 中找不到 <style> 块")
else:
    css_names = set(re.findall(r"\.([a-zA-Z][\w-]*)", sm.group(1)))
    used = set()
    for chunk in (re.findall(r'class="([^"]+)"', js)
                  + re.findall(r"classList\.add\('([^']+)'\)", js)
                  + re.findall(r'classList\.add\("([^"]+)"\)', js)
                  + re.findall(r"className\s*=\s*'([^']*)'", js)
                  + re.findall(r'className\s*=\s*"([^"]*)"', js)):
        for tok in chunk.split():
            if re.fullmatch(r"[a-zA-Z][\w-]*", tok):
                used.add(tok)
    unknown = sorted(used - css_names - {"led", "show", "bad"})
    if unknown:
        errors.append("JS 使用但样式表未定义的 class: %s" % unknown)
    else:
        print("[3] CSS class 校验通过：%d 个类均已定义" % len(used))

# ------------------------------------------------- 4) 下拉取值须服务端支持
# 从 C++ 源码里抓出各枚举参数的合法取值集合
ENUM_SOURCES = {
    # key: (源文件, 抓取函数名)
    "spec": ("product/speculative_options.h", "parse_speculative_backend"),
    "kv_dtype": ("serve/serve_options.cpp", "parse_kv_dtype"),
    "default_reasoning_effort": ("serve/request.h", "parse_requested_reasoning_effort"),
}


def enum_values(path, func):
    """抓取 <func> 里形如 value == "xxx" 的字面量。"""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"%s\s*\([^)]*\)\s*(?:noexcept\s*)?\{" % re.escape(func), text)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    body = text[start:i]
    return set(re.findall(r'==\s*"([^"]+)"', body))


cm = re.search(r"const CHOICES = \{(.*?)\n\};", js, re.S)
if not cm:
    errors.append("index.html 中找不到 CHOICES 定义")
elif not NINFER_SRC.exists():
    print("[4] 跳过下拉取值校验（未找到 ninfer 源码目录）")
else:
    checked = 0
    for key, (rel, func) in ENUM_SOURCES.items():
        got = enum_values(NINFER_SRC / rel, func)
        if got is None:
            errors.append("无法从 %s 的 %s() 抓取枚举取值" % (rel, func))
            continue
        # 取出该 key 在 CHOICES 里的 values 数组
        km = re.search(r"\b%s\s*:\s*\{(.*?)\n  \}" % re.escape(key), cm.group(1), re.S)
        if not km:
            errors.append("CHOICES 中找不到字段 %s" % key)
            continue
        vm = re.search(r"values:\s*\[([^\]]*)\]", km.group(1))
        if not vm:
            errors.append("CHOICES.%s 缺少 values 数组" % key)
            continue
        vals = [v for v in re.findall(r'"([^"]*)"', vm.group(1))]
        bad = [v for v in vals if v != "" and v not in got]
        if bad:
            errors.append("CHOICES.%s 含服务端不支持的值 %s（服务端支持: %s）"
                          % (key, bad, sorted(got)))
        else:
            checked += 1
            print("[4] %s 下拉取值校验通过（服务端支持 %s）" % (key, sorted(got)))

    # vision-residency 是 if/else 链，单独抓
    so = NINFER_SRC / "serve/serve_options.cpp"
    if so.exists():
        t = so.read_text(encoding="utf-8", errors="replace")
        m2 = re.search(r'arg == "--vision-residency"\)\s*\{(.*?)\n        \}', t, re.S)
        if m2:
            supported = set(re.findall(r'mode == "([^"]+)"', m2.group(1)))
            km = re.search(r"\bvision_residency\s*:\s*\{(.*?)\n  \}", cm.group(1), re.S)
            vals = re.findall(r'"([^"]*)"', re.search(r"values:\s*\[([^\]]*)\]", km.group(1)).group(1))
            bad = [v for v in vals if v != "" and v not in supported]
            if bad:
                errors.append("CHOICES.vision_residency 含服务端不支持的值 %s（支持: %s）"
                              % (bad, sorted(supported)))
            else:
                print("[4] vision_residency 下拉取值校验通过（服务端支持 %s）" % sorted(supported))

    # ---- 模板层校验：Qwen3.8 的 chat template 只认 xhigh/medium/low ----
    # 协议层（parse_requested_reasoning_effort）接受 7 个值，但模板会拒绝其余值并
    # 就近映射。如果下拉把模板不支持的值当成"有效档位"呈现，就会误导用户 ——
    # 这正是之前的一个真实缺陷。这里把模板的实际支持集也校验一遍。
    # 注意：chat_templates 与 src 同级（backends/ninfer/tools/...），不在 src 下。
    tpl = NINFER_SRC.parent / "tools" / "chat_templates" / "qwen3_8.jinja"
    if tpl.exists():
        txt = tpl.read_text(encoding="utf-8", errors="replace")
        m3 = re.search(r"not in \(([^)]*)\)", txt)
        if m3:
            tpl_ok = set(re.findall(r"'([^']+)'", m3.group(1)))
            km = re.search(r"\bdefault_reasoning_effort\s*:\s*\{(.*?)\n  \}", cm.group(1), re.S)
            vals = re.findall(r'"([^"]*)"',
                              re.search(r"values:\s*\[([^\]]*)\]", km.group(1)).group(1))
            non_empty = [v for v in vals if v]
            # 下拉里模板不支持的项，其 label 必须显式标注"映射为"，不能伪装成原生档位
            labels = km.group(1)
            mislabeled = []
            for v in non_empty:
                if v in tpl_ok or v == "none":
                    continue
                lm = re.search(r'\b%s\s*:\s*"([^"]*)"' % re.escape(v), labels)
                if not lm or "映射为" not in lm.group(1):
                    mislabeled.append(v)
            if mislabeled:
                errors.append(
                    "default_reasoning_effort 的 %s 不被 Qwen3.8 模板支持，"
                    "但下拉标签未标注『映射为』（模板实际支持: %s）"
                    % (mislabeled, sorted(tpl_ok)))
            else:
                print("[4] reasoning_effort 模板层校验通过"
                      "（模板支持 %s；%s 已标注映射关系）"
                      % (sorted(tpl_ok),
                         sorted(v for v in non_empty if v not in tpl_ok and v != "none")))
        else:
            print("[4] 跳过 reasoning_effort 模板层校验（未找到模板支持集）")

# ---------------------------------------------------------------- 5) 结构
if html.count("<script>") != html.count("</script>"):
    errors.append("<script>/</script> 数量不匹配（%d/%d）"
                  % (html.count("<script>"), html.count("</script>")))
if "__FIELDS__" in html:
    errors.append("index.html 中残留未替换的占位符 __FIELDS__")
if not errors:
    print("[5] 结构校验通过：%d 个 <script> 块，无残留占位符" % html.count("<script>"))

if errors:
    print("\n不一致（FAIL）：")
    for e in errors:
        print("  -", e)
    sys.exit(1)

print("\n全部通过（OK）：前端与 launcher.py 契约一致")
