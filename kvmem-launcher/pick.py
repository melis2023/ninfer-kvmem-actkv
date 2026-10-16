#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""原生文件 / 目录选择框 —— 由 launcher.py 以子进程方式调用，一般不需要手工运行。

为什么独立成子进程（而不是在 HTTP 处理线程里直接弹框）：
  * Tk 需要独占一个主线程和它自己的 Tcl 解释器，放在服务线程里容易互相干扰；
  * 选择框阻塞期间启动器仍能继续响应其它请求（ThreadingHTTPServer 每请求一线程）；
  * 可以单独在 cmd 里跑，便于排查（用法见下）。

用法（launcher.py 用 subprocess 传参，不经过 shell，路径里的空格/中文都无需转义）:

    python pick.py --mode file --title "选择模型文件" --initial "I:\model" ^
        --types "ninfer 模型 (*.ninfer)|*.ninfer|所有文件 (*.*)|*.*"
    python pick.py --mode dir  --title "选择工作目录" --initial "I:\llama"

--types 用 "描述|通配符|描述|通配符" 成对给出（与 Windows 常见写法一致）。
--initial 可以给文件或目录：给文件时自动取其所在目录作为起始目录。

输出（固定 UTF-8）:
    退出码 0  → stdout 一行：所选绝对路径
    退出码 1  → 用户取消（无输出）
    退出码 3  → tkinter 不可用或弹框失败（stderr 一行原因）
"""

import argparse
import os
import sys


def split_types(spec: str):
    """'描述1|通配符1|描述2|通配符2' → [('描述1','通配符1'), ...]"""
    parts = [p for p in (spec or "").split("|") if p != ""]
    return [(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="原生文件/目录选择框（供 launcher.py 调用）")
    ap.add_argument("--mode", choices=("file", "dir"), default="file",
                    help="file=文件选择框；dir=目录选择框")
    ap.add_argument("--title", default="请选择", help="对话框标题")
    ap.add_argument("--initial", default="", help="起始目录（或某个文件，取其所在目录）")
    ap.add_argument("--types", default="", help="文件类型过滤器，见文件头说明")
    a = ap.parse_args(argv)

    # 路径可能含中文，而 Windows 下管道默认编码可能是 cp936 —— 统一强制 UTF-8，
    # 与 launcher.py 读取时使用的 encoding 保持一致。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:                     # 精简版 Python 可能没带 tcl/tk
        print("tkinter 不可用: %s" % e, file=sys.stderr)
        return 3

    initial = (a.initial or "").strip()
    if initial and not os.path.isdir(initial):
        initial = os.path.dirname(initial)
    if not os.path.isdir(initial):
        initial = ""

    try:
        root = tk.Tk()
        root.withdraw()                        # 只要对话框，不要空的主窗口
        root.attributes("-topmost", True)      # 尽量压过浏览器，避免"框跑到后面去了"
        root.update()
        opts = {"title": a.title, "parent": root}
        if initial:
            opts["initialdir"] = initial
        if a.mode == "dir":
            opts["mustexist"] = True
            picked = filedialog.askdirectory(**opts)
        else:
            opts["filetypes"] = split_types(a.types) or [("所有文件", "*.*")]
            picked = filedialog.askopenfilename(**opts)
        try:
            root.destroy()
        except Exception:
            pass
    except Exception as e:
        print("打开选择框失败: %s" % e, file=sys.stderr)
        return 3

    if not picked:
        return 1                               # 用户取消
    print(os.path.abspath(picked))             # abspath 会顺便把 "/" 规范成 "\"
    return 0


if __name__ == "__main__":
    sys.exit(main())
