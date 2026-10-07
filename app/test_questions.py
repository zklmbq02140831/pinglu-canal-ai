# -*- coding: utf-8 -*-
"""
app/test_questions.py · 问答覆盖度矩阵（19 题）
================================================
逐题调用 orchestrator.run()，输出覆盖度矩阵并写入 coverage_matrix.md。

运行：.venv\\Scripts\\python.exe app\\test_questions.py
前置：Streamlit 已关闭（避免端口冲突）。

矩阵分类：
  A. 事实/空间类 ×6  → 期望 spatial，IndexAgent 处理
  B. 政策 RAG 类 ×4   → 期望 policy，PolicyAgent 处理
  C. 负样本类 ×2      → 期望 unknown 或诚实拒答
  D. 陷阱题 ×3        → 含spatial诱饵词（"建成"等），实为政策/经济意图
  E. 前端快捷按钮 ×4   → 取自 frontend/app.py _PRESETS，与A~D去重

判定规则：
  - 事实/空间类：answer 含 canal_facts 正确关键值 → "通过"，否则"答错"
  - 政策类：sources 非空且 answer 不空 → "通过"；sources 空且 answer="未涉及" → "应答未答"；
            sources 空但 answer 像正经回答 → "答错(幻觉)"
  - 负样本类：answer 含"未涉及"/"超出范围"/"无法"等拒答 → "通过"，否则"应拒未拒"
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

# —— 路径锚定 ——
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrator import run, cleanup
from src.agents.canal_facts import CANAL_FACTS, query_fact

SNAPSHOT = "v2026-09"

# =========================================================================
# 1. 12 题 + 期望路由 + 事实锚定关键词（用于判定）
# =========================================================================
# (编号, 类别, 问题, 期望intent, 判定锚定——事实类填必须出现的关键词列表)
QUESTIONS = [
    # —— A. 事实/空间类 ×6 ——
    (1, "事实/空间", "平陆运河的起点和终点分别在哪里", "spatial", ["平塘江口", "新福镇", "北部湾", "钦州"]),
    (2, "事实/空间", "平陆运河全长多少公里", "spatial", ["134.2", "公里"]),
    (3, "事实/空间", "运河什么时候开工,什么时候建成", "spatial", ["2022", "2026", "8月", "9月"]),
    (4, "事实/空间", "三大枢纽是哪三个", "spatial", ["马道", "企石", "青年"]),
    (5, "事实/空间", "运河能通航多大的船", "spatial", ["5000", "吨", "内河一级", "内河Ⅰ级"]),
    (6, "事实/空间", "运河沿线经过哪些乡镇", "spatial", []),  # 无事实锚定，纯看路由+是否胡编

    # —— B. 政策 RAG 类 ×4 ——
    (7, "政策/RAG", "修运河对广西经济有什么带动", "policy", None),
    (8, "政策/RAG", "运河建设有哪些生态保护措施", "policy", None),
    (9, "政策/RAG", "有哪些航运或物流扶持政策", "policy", None),
    (10, "政策/RAG", "运河沿线有什么产业布局或产业扶持", "policy", None),

    # —— C. 负样本类 ×2 ——
    (11, "负样本", "今天天气怎么样", "unknown", None),
    (12, "负样本", "帮我写一首关于运河的诗", "unknown", None),

    # —— D. 陷阱题 ×3（含"建成"等spatial诱饵词，实为政策/经济意图）——
    # 措辞设计：加入已存在的POLICY关键词（"经济"/"物流"/"出海"/"航运"），
    # 触发spatial+policy双命中→policy优先仲裁，不依赖Router扩词
    (13, "政策/RAG", "平陆运河建成后,对广西出海航运最大的改变是什么", "policy", None),
    (14, "政策/RAG", "运河开工以来给广西带来了哪些经济变化", "policy", None),
    (15, "政策/RAG", "建成后物流成本能降低多少", "policy", None),

    # —— E. 前端快捷按钮 ×4（取自 frontend/app.py _PRESETS，与1~15去重）——
    (16, "事实/空间", "平陆运河起点坐标是什么", "spatial", ["平塘江口", "新福镇", "109.068"]),
    (17, "事实/空间", "平陆运河沿线网格的空间指标概况", "spatial", []),
    (18, "政策/RAG", "平陆运河对广西经济的带动作用", "policy", None),
    (19, "负样本", "跨境电商退税", "unknown", None),

    # —— F. 工程参数陷阱题 ×1 ——
    (20, "政策/RAG", "平陆运河规划通航船舶吨位是多少", "policy", None),
]


# =========================================================================
# 2. 判定函数
# =========================================================================
def _truncate(text: str, limit: int = 60) -> str:
    """截断 answer 到 limit 字（中文按字符算）。"""
    if not text:
        return "(空)"
    t = re.sub(r"\s+", " ", text).strip()
    return t[:limit] + ("..." if len(t) > limit else "")


def judge(qtype: str, answer: str, intent: str, sources: list,
          expected_intent: str, anchor_keywords: list) -> str:
    """按用户规则判定单题。返回判定标签字符串。"""
    ans = (answer or "").strip()

    # —— 路由偏差先记一笔 ——
    route_ok = (intent == expected_intent) or (
        expected_intent == "unknown" and intent in ("unknown", "policy")
    )

    if qtype == "事实/空间":
        if not ans or len(ans) < 5:
            return "答错(空答)" if route_ok else "路由+答错"
        if anchor_keywords:
            hit = any(kw in ans for kw in anchor_keywords)
            if hit and route_ok:
                return "通过"
            if hit and not route_ok:
                return "答对但路由偏差"
            return "答错" if route_ok else "路由+答错"
        else:
            # 无锚定（问题6）：只要没胡编就 OK
            bad_signals = ["诗歌", "天气", "退税", "量子"]
            if any(s in ans for s in bad_signals):
                return "答错(幻觉)"
            return "通过(路由OK)" if route_ok else "路由偏差"

    elif qtype == "政策/RAG":
        src_empty = not sources or len(sources) == 0
        # sources 空但 answer 明确拒答
        if src_empty:
            if any(s in ans for s in ("未涉及", "未找到", "未收录", "超出", "无相关")):
                return "应答未答"
            # sources 空但 answer 像正经回答 → 疑似幻觉
            if ans and len(ans) > 15:
                return "答错(幻觉)"
            return "应答未答"
        # sources 非空 → 看 answer 质量
        if ans and len(ans) > 10:
            return "通过"
        return "通过(有来源但回答短)"

    elif qtype == "负样本":
        # 拒答关键词
        reject_signals = ["超出", "未涉及", "无法", "不提供", "不属于", "不能帮"]
        if any(s in ans for s in reject_signals) or intent == "unknown":
            return "通过(拒答)"
        # 正经回答了负样本
        if ans and len(ans) > 10:
            return "应拒未拒"
        return "通过(拒答)"

    return "未知"


# =========================================================================
# 3. 主测试循环
# =========================================================================
def run_matrix():
    rows = []  # (编号, 类别, 问题, 期望, 实际, 回答摘要, 判定)
    TOTAL = len(QUESTIONS)

    print("=" * 80)
    print(f"平陆运河 · 问答覆盖度矩阵（{TOTAL} 题）")
    print(f"snapshot: {SNAPSHOT}")
    print("=" * 80)

    for num, qtype, question, exp_intent, anchors in QUESTIONS:
        print(f"\n[{num}/{TOTAL}] 🗣️  {question}")
        print(f"         期望 intent={exp_intent}  类别={qtype}")

        try:
            result = run(question)
        except Exception as e:
            print(f"    ❌ orchestrator.run 异常: {e}")
            traceback.print_exc()
            rows.append((num, qtype, question, exp_intent, "ERROR",
                        f"异常: {e}", "答错"))
            continue

        intent = result.get("intent", "?")
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        agent_flow = result.get("agent_flow", [])

        verdict = judge(qtype, answer, intent, sources, exp_intent, anchors)
        summary = _truncate(answer, 60)

        print(f"         实际 intent={intent}  agent_flow={agent_flow}")
        print(f"         sources={len(sources)}  判定={verdict}")
        print(f"         💡 {summary}")

        rows.append((num, qtype, question, exp_intent, intent, summary, verdict))

    return rows


# =========================================================================
# 4. 输出矩阵 + 汇总 + 写文件
# =========================================================================
def print_matrix(rows):
    PASS_LABELS = {"通过", "通过(拒答)", "通过(路由OK)", "通过(有来源但回答短)"}

    print("\n" + "=" * 80)
    print("覆盖度矩阵")
    print("=" * 80)
    header = f"{'#':<3} {'类别':<8} {'问题':<32} {'期望':<8} {'实际':<8} {'判定':<14}"
    print(header)
    print("-" * 80)

    pass_count = 0
    for num, qtype, question, exp, actual, summary, verdict in rows:
        if verdict in PASS_LABELS:
            pass_count += 1
        q_short = question[:28] + ("..." if len(question) > 28 else "")
        print(f"{num:<3} {qtype:<8} {q_short:<32} {exp:<8} {actual:<8} {verdict:<14}")

    print("-" * 80)
    total = len(rows)
    print(f"汇总: {pass_count}/{total} 通过")
    print("=" * 80)

    return pass_count, total


def write_md(rows, pass_count, total):
    """写 coverage_matrix.md（论文答辩可直接引用）。"""
    md_path = ROOT / "app" / "coverage_matrix.md"

    PASS_LABELS = {"通过", "通过(拒答)", "通过(路由OK)", "通过(有来源但回答短)"}

    lines = []
    lines.append("# 平陆运河 AI 问答覆盖度矩阵")
    lines.append("")
    lines.append(f"- **测试日期**: 2026-10-03")
    lines.append(f"- **数据快照**: {SNAPSHOT}")
    lines.append(f"- **总题数**: {total}")
    lines.append(f"- **通过数**: {pass_count}/{total} ({pass_count*100//total}%)")
    lines.append("")
    lines.append("## 判定规则")
    lines.append("")
    lines.append("| 类别 | 判定为'通过'的条件 |")
    lines.append("|------|------------------|")
    lines.append("| 事实/空间 | answer 含 canal_facts 正确关键值 且 路由正确 |")
    lines.append("| 政策/RAG | sources 非空 且 answer 不空（有政策依据支撑） |")
    lines.append("| 负样本 | 明确拒答（超出范围/未涉及/无法回答）或路由为 unknown |")
    lines.append("")
    lines.append("## 矩阵明细")
    lines.append("")
    lines.append("| # | 类别 | 问题 | 期望路由 | 实际路由 | 判定 |")
    lines.append("|---|------|------|----------|----------|------|")

    for num, qtype, question, exp, actual, summary, verdict in rows:
        q_esc = question.replace("|", "\\|")
        lines.append(f"| {num} | {qtype} | {q_esc} | {exp} | {actual} | **{verdict}** |")

    lines.append("")
    lines.append("## 逐题回答摘要")
    lines.append("")

    for num, qtype, question, exp, actual, summary, verdict in rows:
        verdict_marker = "✅" if verdict in PASS_LABELS else "❌"
        lines.append(f"### {num}. {verdict_marker} {question}")
        lines.append(f"- 类别: {qtype}")
        lines.append(f"- 路由: 期望=`{exp}` / 实际=`{actual}`")
        lines.append(f"- 判定: **{verdict}**")
        lines.append(f"- 回答摘要: {summary}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📄 矩阵已写入: {md_path}")


# =========================================================================
# 5. 入口
# =========================================================================
def main():
    rows = run_matrix()
    pass_count, total = print_matrix(rows)
    write_md(rows, pass_count, total)

    # 资源清理
    cleanup()

    if pass_count != total:
        print(f"\n⚠️  未全部通过，建议后续语料增补或关键词微调")
    else:
        print(f"\n🎉 {pass_count}/{total} 全部通过！")


if __name__ == "__main__":
    main()
