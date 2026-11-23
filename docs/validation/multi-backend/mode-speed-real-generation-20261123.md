# 三模式真实生成测速：鹈鹕骑车 HTML（2026-11-23）

前一轮 128k 重复行测速只压 ngram 复制路径（batch=1 开 ngram 时神经草稿前向整段跳过），
测不到草稿模型本身。本轮换成真实创作型请求，让三种推测后端真正走各自的草稿网络。

## 1. 请求与条件

- 提示词：`制作一个鹈鹕骑车的 html，画面美观，场景丰富`（裸用户回合 27 token，加 system 后 prompt 共 82 token）
- 外加一段 system：只输出单个完整 HTML、不解释、不调工具、以 `</html>` 结尾。
  该工件带 agent 微调痕迹，只给裸用户回合时会去调 `list_files` / `read_file`，
  首轮试跑正是因此只出了 108 token 就 stop，所以必须约束。
- 采样：`temperature 0`、`reasoning_effort none`、`max_tokens 16384`（首轮 8192 会在
  20,056 字符处被截断，调到 16384 后 DFlash2/DSpine 自然收尾）。
- 服务参数：两模式 `--ngram-draft-tokens 12 --draft-tokens 4 --kv-dtype rk8v4`，
  DSpine 草稿 KV 必须 BF16（`--draft-kv-dtype bf16`），MTP 用同窗口 + `--adaptive-mtp
  --full-draft-head`。三模式共用同一工件与同一 prompt，逻辑上只有草稿后端不同。
- 每轮单独重启服务（重新加载模型），交错顺序 dflash2 → dspine → mtp，共两轮。

## 2. 结果

| 轮 | 模式 | 输出 token | wall | decode | 接受率 | ngram 复制轮 |
|---|---|---:|---:|---:|---:|---:|
| c1 | DFlash2 | 10,310 | 124.3 s | 85.3 tok/s | 7,576/11,780（64.3%） | 106 |
| c1 | DSpine | 11,920 | 125.8 s | 97.7 tok/s | 9,152/12,121（**75.5%**） | 132 |
| c1 | MTP | 16,384（触顶） | 161.1 s | 104.4 tok/s | 13,367/17,030（78.5%） | 665 |
| c2 | DFlash2 | 10,310 | 120.9 s | 87.9 tok/s | 7,576/11,780（64.3%） | 106 |
| c2 | DSpine | 11,920 | 122.2 s | 100.6 tok/s | 9,152/12,121（**75.5%**） | 132 |
| c2 | MTP | 12,572 | 164.2 s | 78.5 tok/s | 9,264/14,517（63.8%） | 201 |

两轮中位 decode：**DSpine 99.2 > MTP 91.5 > DFlash2 86.6 tok/s**。

- DFlash2 / DSpine 两轮**逐字节可复现**（10,310 与 11,920 token，26,371 与 28,067 字符
  两次完全一致），说明这两个后端的验证路径没有引入随机性，测量噪声只剩 GPU 状态
  （85.3 → 87.9、97.7 → 100.6，差 3%）。
- MTP 两轮**不可复现**（16,384 触顶 vs 12,572 自然收尾）。`--adaptive-mtp` 的自适应草稿
  长度会改变每轮提议宽度，进而改变采样路径，temp 0 也压不住。所以 MTP 那两行只能当参考，
  不能当结论。

## 3. 差距来自接受率，不是「哪个模式更快」

12k 上下文下每轮的目标前向成本三模式相同（同一工件、同一 verify 宽度），所以 decode 的
差距几乎全部落在草稿接收率上：

| 模式 | 接受率 | decode |
|---|---:|---:|
| DSpine | 75.5% | 97.7–100.6 |
| DFlash2 | 64.3% | 85.3–87.9 |

11.2 个百分点的接受率差，对应约 14% 的 decode 差。这与 128k 那一轮的结论一致：
**同窗口下模式间差异由接受率和 GPU 状态决定，不存在固定的「某模式快 30%」。**

另外注意 ngram 在创作型文本上的命中率远低于重复行场景：DFlash2 只复制出 652/1,272
（51%），而 128k 重复行测试是 434/434（100%）。所以这轮才是草稿网络真实能力的体现，
也是 DSpine 环按绝对位置写入、接受率更高的价值所在。

## 4. 输出正确性

DFlash2 / DSpine 的四份输出都是完整单文件 HTML（1 个 `<style>`、1 个 `<script>`、
5 个 `<svg>`，以 `</html>` 结尾）。MTP c1 因触顶 16,384 token 在 SVG path 中间被截断，
不计入正确性；MTP c2 完整。

三模式在同一提示下字符级分歧出现在第 267 字符（`"Bodoni MT"` vs `"Bodoni 72"` 字体名），
属于采样路径差异，不是结构性问题；同模式跨轮则完全一致。

样例见 [pelican-bike-dspine-20261123.html](samples/pelican-bike-dspine-20261123.html)
（DSpine，11,920 token / 28,067 字符 / 2m05.8s）。

## 5. 复现

`C:\temp\dspine_bench\bench_pelican.py [cycles] [modes]`：起服务 → 等就绪 → 发请求 →
落 `pelican/<mode>_c<n>.html` → 杀进程。结果汇总写 `result_pelican.json`，
服务日志 `srv_pelican_c<n>_<mode>.log`。

## 6. 限制

- 单提示词、单工件、单机（RTX 5060 Ti 16 GB）。接受率强依赖文本内容，换题材可能换排序。
- 只有两轮，MTP 又不可复现；要下定论需要 MTP 单独多跑 4–5 轮。
- 上下文最长约 12k token，未覆盖 128k 档；那一档见
  [ngram 窗口 128k 实测](../ngram/ngram-window-128k-validation-20261123.md)。
- 未测工具调用路径：DSpine 在结构化输出上仍有独立的
  `generated token 27 violates the structured output grammar` 问题。
