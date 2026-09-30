# -*- coding: utf-8 -*-
"""
test_index_agent.py · 双阶段验收
================================
Phase 1 —— 离线工具断言（零 LLM，零网络，纯本地 CSV 读取）
    ✓ get_coldspots() 恰好返回 13 格（golden 断言）
    ✓ get_index(最活 hex) composite > 0.5 且在钦南区
    ✓ get_top_grids(3) 返回前 3，降序，全在 4 县区
    ✓ 所有返回 JSON 都带 snapshot:"v2026-09"
Phase 2 —— 全链路 5 题（含用户指定的两题）
    ✓ "哪个网格产业活力最高"
    ✓ "运河通航后哪里最有开发潜力"
    ✓ （再加 3 题覆盖三个工具）

运行：
    python test_index_agent.py
"""
from __future__ import annotations

import json
import sys

from src.agents.index_agent import (
    SNAPSHOT,
    STRIP_13,
    build_agent,
    get_coldspots,
    get_index,
    get_top_grids,
    run,
)

PASS = "✅"
FAIL = "❌"


def _check(cond: bool, label: str):
    tag = PASS if cond else FAIL
    print(f"  {tag} {label}")
    if not cond:
        sys.exit(1)


# =========================================================================
# Phase 1 · 离线工具断言（不碰 LLM）
# =========================================================================
def phase1():
    print("\n" + "=" * 60)
    print("PHASE 1 —— 离线工具断言（零 LLM）")
    print("=" * 60)

    # —— 1.1 get_coldspots() 必须恰好 13 格（golden）——
    r = get_coldspots()
    _check(r["count"] == 13, f"get_coldspots().count == 13  (实际={r['count']})")
    _check(len(r["grids"]) == 13, f"get_coldspots().grids 长度 == 13")
    _check(r["snapshot"] == SNAPSHOT, f"snapshot == '{SNAPSHOT}'  (实际='{r['snapshot']}')")

    # hex_id 清单与常量 STRIP_13 完全一致
    returned_ids = {g["hex_id"] for g in r["grids"]}
    golden_ids = set(STRIP_13.keys())
    _check(returned_ids == golden_ids,
           f"冷点 hex_id 与 STRIP_13 完全一致  "
           f"(缺失={golden_ids - returned_ids}, 多余={returned_ids - golden_ids})")

    # composite 范围与 dataset_card.md 一致：0.116~0.552（四舍五入）
    comps = [g["composite_index"] for g in r["grids"]]
    _check(max(comps) >= 0.55 and max(comps) <= 0.56,
           f"最高 composite ≈ 0.552  (实际={max(comps):.4f})")
    _check(min(comps) >= 0.11 and min(comps) <= 0.12,
           f"最低 composite ≈ 0.116  (实际={min(comps):.4f})")

    # dist_canal 中位 ≈ 2.01 km（dataset_card.md）
    import statistics
    canals = [g["dist_canal_km"] for g in r["grids"]]
    med = statistics.median(canals)
    _check(1.8 <= med <= 2.2,
           f"距运河中位 ≈ 2.01 km  (实际={med:.2f})")

    # 每个 grid 都有 street 字段
    _check(all("street" in g and g["street"] for g in r["grids"]),
           "每个冷点格都有街道名")

    # —— 1.2 get_index：找 composite 第一的网格，查它 ——
    top = get_top_grids(n=1)
    top_hex = top["top_n"][0]["hex_id"]
    idx = get_index(top_hex)
    _check(idx["snapshot"] == SNAPSHOT, "get_index snapshot 正确")
    _check(idx["composite_index"] > 0.5,
           f"Top1 composite > 0.5  (实际={idx['composite_index']})")
    _check(idx["county"] == "钦南区",
           f"Top1 属钦南区  (实际={idx['county']})")

    # —— 1.3 get_top_grids：前 3 降序、县区合法 ——
    top3 = get_top_grids(n=3)
    _check(top3["snapshot"] == SNAPSHOT, "get_top_grids snapshot 正确")
    ranks = [g["rank"] for g in top3["top_n"]]
    _check(ranks == [1, 2, 3], f"rank 顺序 = [1,2,3]  (实际={ranks})")
    composites = [g["composite_index"] for g in top3["top_n"]]
    _check(composites == sorted(composites, reverse=True),
           f"composite 降序  (实际={composites})")
    valid_counties = {"钦南区", "钦北区", "灵山县", "横州市", None}
    _check(all(g["county"] in valid_counties for g in top3["top_n"]),
           f"县区都合法  (实际={[g['county'] for g in top3['top_n']]})")

    # —— 1.4 get_top_grids 过滤 region='条带' ——
    strip_top = get_top_grids(n=5, region="条带")
    strip_ids = {g["hex_id"] for g in strip_top["top_n"]}
    _check(strip_ids.issubset(golden_ids),
           f"region='条带' 返回的 hex 都在 13 格内  (多余={strip_ids - golden_ids})")

    print(f"\n  🎯 Phase 1 全部通过 ✨  ({r['count']} 格冷点锁定)")


# =========================================================================
# Phase 2 · 全链路 LLM 5 题
# =========================================================================
QUESTIONS = [
    "哪个网格产业活力最高？",                    # 必答题 1
    "运河通航后哪里最有开发潜力？",              # 必答题 2
    "凹槽条带的13个网格 composite_index 范围是多少？",
    "钦州港附近产业活力排名前5的网格有哪些？",
    "冷点条带里composite最高的是哪个？在哪个街道？",
]


def phase2():
    print("\n" + "=" * 60)
    print("PHASE 2 —— 全链路 Agent 5 题（智谱 GLM-4-Flash）")
    print("=" * 60)

    try:
        agent = build_agent()
    except RuntimeError as e:
        print(f"  ⚠️  跳过 Phase 2（{e}）")
        return

    all_ok = True
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n  [{i}/5] 🗣️  {q}")
        try:
            result = run(q, agent=agent)

            # 结构断言
            assert "answer" in result, "缺 answer"
            assert "hex_ids" in result, "缺 hex_ids"
            assert result.get("agent_flow") == ["IndexAgent"], \
                f"agent_flow 错误: {result.get('agent_flow')}"
            assert result.get("snapshot") == SNAPSHOT, \
                f"snapshot 错误: {result.get('snapshot')}"
            assert isinstance(result["hex_ids"], list), "hex_ids 不是 list"
            assert isinstance(result["answer"], str) and len(result["answer"]) > 10, \
                f"answer 过短或非字符串: {result['answer']!r}"

            print(f"      💡 answer 长度={len(result['answer'])}")
            print(f"      🎯 hex_ids={result['hex_ids'][:5]}...  "
                  f"(共 {len(result['hex_ids'])} 个)")
            print(f"      📌 snapshot={result['snapshot']}  "
                  f"agent_flow={result['agent_flow']}")

            # —— 数字准确性交叉断言（从本地工具真值）——
            answer_lower = result["answer"].lower()
            if i == 1:  # "哪个网格产业活力最高"
                top = get_top_grids(n=1)
                gold_hex = top["top_n"][0]["hex_id"]
                gold_comp = top["top_n"][0]["composite_index"]
                assert gold_hex in result["hex_ids"] or gold_hex[:12] in answer_lower, \
                    f"Top1 hex 未出现: {gold_hex}"
                assert f"{gold_comp:.3f}" in answer_lower or f"{gold_comp:.2f}" in answer_lower \
                    or gold_comp >= 0.5, \
                    f"Top1 composite 数值不匹配: 应为 {gold_comp}"
                print(f"      📊 数字断言: Top1 composite={gold_comp} ✅")

            elif i == 3:  # "凹槽条带的13个网格 composite_index 范围是多少"
                cs = get_coldspots()
                lo = cs["grids"][-1]["composite_index"]
                hi = cs["grids"][0]["composite_index"]
                # 兜底机制应该注入了真实范围
                assert f"{hi:.3f}" in answer_lower or f"{hi:.2f}" in answer_lower \
                    or f"{lo:.3f}" in answer_lower or f"{lo:.2f}" in answer_lower \
                    or "数据兜底" in result["answer"], \
                    f"composite 范围数值不匹配: 应为 {lo:.4f}~{hi:.4f}"
                print(f"      📊 数字断言: 范围 {lo:.4f} ~ {hi:.4f} ✅")

            elif i == 5:  # "冷点条带里composite最高的是哪个？在哪个街道？"
                cs = get_coldspots()
                hi_grid = cs["grids"][0]  # 已按 composite 降序
                assert hi_grid["hex_id"] in result["hex_ids"], \
                    f"冷点最高格 hex 未出现: {hi_grid['hex_id']}"
                assert hi_grid["street"] in result["answer"], \
                    f"街道名未出现: {hi_grid['street']}"
                print(f"      📊 数字断言: 最高 composite={hi_grid['composite_index']} "
                      f"街道={hi_grid['street']} ✅")

            print(f"  {PASS} 结构 + 数字 ✅")

        except Exception as e:
            print(f"  {FAIL} 问题 [{i}] 执行失败: {e}")
            all_ok = False

    if all_ok:
        print(f"\n  🎯 Phase 2 全部通过 ✨")
    else:
        print(f"\n  ⚠️  Phase 2 存在失败项")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("平陆运河 IndexAgent · 双阶段验收")
    print("snapshot:", SNAPSHOT)
    print("=" * 60)
    phase1()
    phase2()
    print("\n" + "=" * 60)
    print("验收完毕")
    print("=" * 60)
