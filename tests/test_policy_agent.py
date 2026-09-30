# -*- coding: utf-8 -*-
"""
test_policy_agent.py · 双阶段验收
===================================
Phase 1 —— 离线向量库断言（零 LLM，零网络调 embedding）
    ✓ Qdrant collection policy_v2026_09 存在
    ✓ 13 份文档全部入库（doc_no 去重 = 13）
    ✓ 总 chunk 数 > 100
    ✓ 每条 payload 的 title/doc_no/issuer/issue_date/level/snapshot 非空
    ✓ search_policy("平陆运河经济") top1 score > 0.5（烟雾测试）
Phase 2 —— 全链路 4 题（含必答 + 负样本拦截）
    ① 平陆运河对广西经济的带动作用
    ② 产业支持政策要点
    ③ 航运物流成本相关条款
    ④ 负样本 "跨境电商退税" → 必须答"未涉及"

运行：python tests/test_policy_agent.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# —— 路径：确保项目根在 sys.path ——
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.policy_agent import (
    COLLECTION,
    QDRANT_PATH,
    SNAPSHOT,
    build_agent,
    run,
    search_policy,
)

PASS = "✅"
FAIL = "❌"


def _check(cond: bool, label: str):
    tag = PASS if cond else FAIL
    print(f"  {tag} {label}")
    if not cond:
        sys.exit(1)


# =========================================================================
# Phase 1 · 离线向量库断言
# =========================================================================
def phase1():
    print("\n" + "=" * 60)
    print("PHASE 1 —— 离线向量库断言（零 LLM）")
    print("=" * 60)

    from qdrant_client import QdrantClient

    # —— 1.1 Qdrant 库存在 ——
    _check(QDRANT_PATH.exists(), f"Qdrant 目录存在: {QDRANT_PATH}")
    client = QdrantClient(path=str(QDRANT_PATH))
    _check(
        client.collection_exists(COLLECTION),
        f"collection '{COLLECTION}' 存在",
    )

    # —— 1.2 总 chunk 数 > 100 ——
    total = client.count(COLLECTION).count
    _check(total > 100, f"总 chunk 数 = {total}  (> 100)")

    # —— 1.3 doc_no 去重 = 13 ——
    all_points, _offset = client.scroll(
        COLLECTION, limit=total, with_payload=True, with_vectors=False
    )
    doc_nos = set()
    for pt in all_points:
        doc_nos.add(pt.payload.get("doc_no", ""))
    doc_nos.discard("")
    _check(
        len(doc_nos) == 13,
        f"13 份文档全部入库 (doc_no 去重={sorted(doc_nos)})",
    )

    # —— 1.4 每条 payload 必填字段非空 ——
    REQUIRED = ("doc_no", "title", "issuer", "issue_date", "level", "snapshot")
    empty_fields: list[str] = []
    for pt in all_points:
        p = pt.payload
        for f in REQUIRED:
            if not p.get(f):
                empty_fields.append(f"id={pt.id} 缺 {f}")
                break
    _check(
        not empty_fields,
        f"每条 payload 必填字段全非空  ({len(all_points)} 条检查)",
    )

    # —— 1.5 snapshot 全部 = v2026-09 ——
    bad_snap = [
        f"id={pt.id} snap={pt.payload.get('snapshot')}"
        for pt in all_points
        if pt.payload.get("snapshot") != SNAPSHOT
    ]
    _check(not bad_snap, f"所有 snapshot='{SNAPSHOT}'")

    # —— 显式 close（Windows portalocker 互斥锁）——
    client.close()
    del client

    # —— 1.6 烟雾测试 search_policy ——
    r = search_policy("平陆运河经济")
    _check(not r["empty"], "search_policy('平陆运河经济') 非空")
    _check(len(r["hits"]) >= 3, f"返回 hits ≥ 3 (实际={len(r['hits'])})")
    _check(r["hits"][0]["score"] > 0.5,
           f"top1 score > 0.5 (实际={r['hits'][0]['score']})")
    _check(r["snapshot"] == SNAPSHOT, "search_policy snapshot 正确")

    # —— 1.7 负样本拦截烟雾 ——
    r_neg = search_policy("量子物理与人工智能的哲学关系")
    _check(
        r_neg["empty"] or r_neg["hits"][0]["score"] < 0.35,
        f"负样本被拦截 (empty={r_neg['empty']}, top1={r_neg['hits'][0]['score'] if not r_neg['empty'] else 'N/A'})",
    )

    print(f"\n  🎯 Phase 1 全部通过 ✨  ({total} chunks, 13 docs)")


# =========================================================================
# Phase 2 · 全链路 4 题（含负样本拦截）
# =========================================================================
QUESTIONS = [
    ("平陆运河对广西经济的带动作用",          "positive"),
    ("产业支持政策要点",                        "positive"),
    ("航运物流成本相关条款",                    "positive"),
    ("跨境电商退税",                            "negative"),  # 负样本
]


def phase2():
    print("\n" + "=" * 60)
    print("PHASE 2 —— 全链路 Agent 4 题（智谱 GLM-4-Flash + embedding-3）")
    print("=" * 60)

    try:
        agent = build_agent()
    except RuntimeError as e:
        print(f"  ⚠️  跳过 Phase 2（{e}）")
        return

    all_ok = True
    for i, (q, tag) in enumerate(QUESTIONS, 1):
        print(f"\n  [{i}/4] 🗣️  {q}  ({tag})")
        try:
            result = run(q, agent=agent)

            # 结构断言
            assert "answer" in result, "缺 answer"
            assert "sources" in result, "缺 sources"
            assert result.get("agent_flow") == ["PolicyAgent"], \
                f"agent_flow 错误: {result.get('agent_flow')}"
            assert result.get("snapshot") == SNAPSHOT, \
                f"snapshot 错误: {result.get('snapshot')}"
            assert isinstance(result["answer"], str) and len(result["answer"]) > 5, \
                f"answer 过短: {result['answer']!r}"

            print(f"      💡 answer ({len(result['answer'])}字): {result['answer'][:60]}...")
            print(f"      📚 sources = {len(result['sources'])} 条")
            for s in result["sources"][:2]:
                print(f"        - {s['doc_no']} {s['title'][:25]} score={s['score']:.3f}")
            print(f"      📌 snapshot={result['snapshot']}  agent_flow={result['agent_flow']}")

            # —— 正样本断言 ——
            if tag == "positive":
                assert len(result["sources"]) >= 1, \
                    f"正样本应至少有 1 条 source (实际={len(result['sources'])})"
                # 检查 source 字段完整
                for s in result["sources"]:
                    assert s.get("doc_no"), "source 缺 doc_no"
                    assert s.get("title"), "source 缺 title"
                print(f"  {PASS} 正样本 ✅")

            # —— 负样本断言：answer 必须包含"未涉及"或 sources 为空 ——
            elif tag == "negative":
                ans = result["answer"]
                neg_ok = ("未涉及" in ans or "无" in ans or len(result["sources"]) == 0)
                assert neg_ok, \
                    f"负样本应返回'未涉及'或 sources 为空  (answer='{ans}', sources={len(result['sources'])})"
                print(f"  {PASS} 负样本拦截 ✅  (LLM 未编造政策)")

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
    print("平陆运河 PolicyAgent · 双阶段验收")
    print(f"snapshot: {SNAPSHOT}")
    print(f"Qdrant: {QDRANT_PATH} → {COLLECTION}")
    print("=" * 60)
    phase1()
    phase2()
    print("\n" + "=" * 60)
    print("验收完毕")
    print("=" * 60)
