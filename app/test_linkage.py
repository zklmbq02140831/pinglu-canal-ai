# -*- coding: utf-8 -*-
"""
app/test_linkage.py · RouterAgent + Orchestrator 联动测试（6 案例）
==================================================================
纯 Python 断言脚本，运行：python app/test_linkage.py

案例覆盖：
  ① 空间题 ×2    → intent=spatial, agent_flow 含 IndexAgent
  ② 政策题 ×2    → intent=policy, 其中"跨境电商退税"(负样本) sources=[]
  ③ 混合题 ×1    → 同时命中 spatial/policy 关键词，按稳定策略 → policy
  ④ 无关题 ×1    → intent=unknown, 不调用下游 Agent
附加断言：agent_flow[0]=="Router"；所有返回含 snapshot 字段。
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# —— 路径：确保项目根在 sys.path ——
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrator import run, cleanup

SNAPSHOT = "v2026-09"
PASS = "✅"
FAIL = "❌"


def _check(cond: bool, label: str, detail: str = ""):
    tag = PASS if cond else FAIL
    extra = f"  ({detail})" if detail and not cond else ""
    print(f"    {tag} {label}{extra}")
    if not cond:
        raise AssertionError(f"{label}{': ' + detail if detail else ''}")


def test_case(i: int, question: str, expected_intent: str,
              must_contain_agent: str = None,
              expect_sources_empty: bool = False):
    """执行单个测试案例，返回 (ok: bool, result: dict)。"""
    print(f"\n  [{i}/6] 🗣️  {question}")
    print(f"         期望 intent={expected_intent}" +
          (f", 必须含 {must_contain_agent}" if must_contain_agent else ""))

    try:
        result = run(question)
    except Exception as e:
        print(f"    {FAIL} orchestrator.run 抛出异常: {e}")
        traceback.print_exc()
        return False, {}

    # —— 基础结构断言 ——
    _check(isinstance(result, dict), "返回值为 dict")
    _check("answer" in result, "含 answer 字段")
    _check("intent" in result, "含 intent 字段")
    _check("router_reason" in result, "含 router_reason 字段")
    _check("agent_flow" in result, "含 agent_flow 字段")
    _check("snapshot" in result, "含 snapshot 字段")

    # —— snapshot 必须正确 ——
    _check(result["snapshot"] == SNAPSHOT,
           f"snapshot='{result['snapshot']}'",
           f"期望 '{SNAPSHOT}'")

    # —— agent_flow[0] 必须是 Router ——
    _check(len(result["agent_flow"]) >= 1, "agent_flow 非空")
    _check(result["agent_flow"][0] == "Router",
           f"agent_flow[0]='{result['agent_flow'][0]}'",
           "期望 'Router'")

    # —— intent 断言 ——
    _check(result["intent"] == expected_intent,
           f"intent='{result['intent']}'",
           f"期望 '{expected_intent}'")

    # —— 下游 Agent 断言（unknown 无下游 Agent）——
    if expected_intent != "unknown":
        _check(len(result["agent_flow"]) >= 2,
               "agent_flow 含 Router + 下游 Agent",
               f"实际 agent_flow={result['agent_flow']}")
        if must_contain_agent:
            _check(must_contain_agent in result["agent_flow"],
                   f"agent_flow 含 {must_contain_agent}",
                   f"实际 agent_flow={result['agent_flow']}")
    else:
        # unknown 不应调用任何下游 Agent
        _check(len(result["agent_flow"]) == 1 and result["agent_flow"][0] == "Router",
               "unknown 仅含 Router 无下游 Agent",
               f"实际 agent_flow={result['agent_flow']}")

    # —— 负样本 sources 断言 ——
    if expect_sources_empty:
        _check("sources" in result, "含 sources 字段")
        _check(len(result["sources"]) == 0,
               "负样本 sources 为空",
               f"实际 len(sources)={len(result.get('sources', []))}")

    # —— answer 非空断言（unknown 除外，unknown 的 answer 是固定提示）——
    _check(isinstance(result["answer"], str) and len(result["answer"]) > 0,
           "answer 非空字符串",
           f"实际 answer={result.get('answer')!r}")

    # —— 打印结果摘要 ——
    ans_preview = result["answer"][:50] + "..." if len(result["answer"]) > 50 else result["answer"]
    print(f"         💡 answer({len(result['answer'])}字): {ans_preview}")
    print(f"         📌 intent={result['intent']}  "
          f"agent_flow={result['agent_flow']}  "
          f"snapshot={result['snapshot']}")
    print(f"         🔍 reason: {result['router_reason'][:80]}")

    return True, result


# =========================================================================
# 主测试
# =========================================================================
def main():
    print("=" * 65)
    print("平陆运河 · RouterAgent + Orchestrator 联动测试（6 案例）")
    print(f"snapshot: {SNAPSHOT}")
    print("=" * 65)

    ALL_CASES = [
        # (问题, 期望 intent, 必须含哪个 Agent, sources 是否必须空)
        ("平陆运河起点坐标是什么",          "spatial", "IndexAgent", False),  # 空间题1
        ("检索沿线5公里内的乡镇",            "spatial", "IndexAgent", False),  # 空间题2
        ("平陆运河对广西经济的带动作用",      "policy",  "PolicyAgent", False),  # 政策题1
        ("跨境电商退税",                      "policy",  "PolicyAgent", True),   # 政策题2（负样本）
        ("运河沿线的产业扶持政策有哪些",      "policy",  "PolicyAgent", False),  # 混合题（policy 优先）
        ("今天天气怎么样",                    "unknown", None, False),            # 无关题
    ]

    results = []
    for i, (q, exp_intent, must_agent, empty_src) in enumerate(ALL_CASES, 1):
        ok, _ = test_case(i, q, exp_intent, must_agent, empty_src)
        results.append((i, ok, q))

    # —— 汇总 ——
    print("\n" + "=" * 65)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"结果汇总：{PASS if passed == total else FAIL} {passed}/{total} 通过")
    for i, ok, q in results:
        tag = PASS if ok else FAIL
        print(f"  {tag} [{i}] {q}")
    print("=" * 65)

    if passed != total:
        sys.exit(1)
    print("🎉 全部通过")

    # —— 资源清理：关闭 QdrantClient 等懒加载 client ——
    cleanup()


if __name__ == "__main__":
    main()
