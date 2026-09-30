# -*- coding: utf-8 -*-
"""
build_policy_vector_store.py · 离线向量化批处理
===================================================
读取 13 篇政策 txt → 按章节/条款切 chunk → 调智谱 embedding-3 → 写 Qdrant 本地库。

Pipeline：
    data/processed/docs_clean/*.txt       (13 篇清洗后政策正文)
    data/processed/docs_clean/docs_meta/*.json (per-file 元数据)
        ↓ chunking + embedding
    data/qdrant/                          (QdrantClient(path=...))
    collection: policy_v2026_09

Payload schema（每条 vector 必带）：
    {doc_no, title, issuer, issue_date, level, chunk_index, chunk_text, snapshot:"v2026-09"}

运行（离线批处理，一次性）：
    .venv\\Scripts\\python.exe scripts/build_policy_vector_store.py

验收：
    - 13 篇全部入库
    - 总 chunk 数 > 100
    - 每条 payload 上述字段全非空
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# —— 路径锚定（脚本约定）——
ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "processed" / "docs_clean"
META_DIR = ROOT / "data" / "processed" / "docs_meta"    # ← meta JSON 真实位置
CATALOG_JSON = META_DIR / "docs_catalog.json"            # 含"效力级别"
QDRANT_PATH = ROOT / "data" / "qdrant"
COLLECTION = "policy_v2026_09"
SNAPSHOT = "v2026-09"

# —— 智谱 embedding 配置 ——
EMBED_MODEL = "embedding-3"
EMBED_DIM = 2048  # embedding-3 输出维度
EMBED_BATCH = 10  # 每批 embedding 调用数（避免 429）


# =========================================================================
# 0. Key 读取（多路径兜底）
# =========================================================================
def _read_zhipu_key() -> str:
    candidates = [
        p for p in [
            ROOT / ".env",
            Path(r"D:\yunhe\.env"),
            Path(r"d:\112\yunhe\.env"),
        ]
        if p.exists()
    ]
    for p in candidates:
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("ZHIPU_API_KEY", "ZHIPUAI_API_KEY") and v.strip():
                return v.strip().strip("'\"")
    return ""


# =========================================================================
# 1. 切 chunk：章节/条款 → 段落
# =========================================================================
def _num(name: str) -> str:
    return name.split("_", 1)[0]


def _normalize_spaces(text: str) -> str:
    """清洗 Unicode 空格/全角空格/零宽字符，智谱 embedding API 对这些敏感。"""
    import unicodedata
    # 全角空格 → 半角空格
    text = text.replace("\u3000", " ")
    # Unicode 空格类（\u2000-\u200A, \u202F, \u205F, \u00A0）→ 普通空格
    text = re.sub(r"[\u2000-\u200A\u202F\u205F\u00A0]", " ", text)
    # 零宽字符清理
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    # 多余空白折叠
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_into_chunks(text: str, doc_no: str) -> list[tuple[int, str]]:
    """
    返回 [(chunk_index, chunk_text), ...]
    策略（按优先级）：
        ① 有"第X条"（法规/条例类）→ 按条切
        ② 有"第X章"（纲要类）→ 按章切
        ③ 有中文顿号"X、"（规划类）→ 按顿号切
        ④ 总字数 < 5000 → 整篇 1 chunk
        ⑤ 都不命中 → 按段落（\n\n）硬切
    """
    # ① 按条切（如 01/02 法规类）
    art_pat = re.compile(r"^第[一二三四五六七八九十百零\d]+条\b", re.M)
    arts = list(art_pat.finditer(text))
    if arts:
        chunks: list[str] = []
        for i, m in enumerate(arts):
            start = m.start()
            end = arts[i + 1].start() if i + 1 < len(arts) else len(text)
            body = text[start:end].strip()
            if body:
                chunks.append(body)
        # 连续短条款（<120字）合并 → 避免碎块
        merged: list[str] = []
        for c in chunks:
            if merged and len(c) < 120:
                merged[-1] += "\n" + c
            else:
                merged.append(c)
        return list(enumerate(merged))

    # ② 按章切（如 06 十五五纲要）
    chap_pat = re.compile(r"^第[一二三四五六七八九十百零\d]+章\b", re.M)
    chaps = list(chap_pat.finditer(text))
    if chaps:
        chunks = []
        for i, m in enumerate(chaps):
            start = m.start()
            end = chaps[i + 1].start() if i + 1 < len(chaps) else len(text)
            body = text[start:end].strip()
            if body:
                chunks.append(body)
        return list(enumerate(_split_long_chunks(chunks, 1500)))

    # ③ 按中文顿号切（如 03 先进制造业规划）
    num_pat = re.compile(r"^[一二三四五六七八九十百零]+、", re.M)
    nums = list(num_pat.finditer(text))
    if nums:
        chunks = []
        for i, m in enumerate(nums):
            start = m.start()
            end = nums[i + 1].start() if i + 1 < len(nums) else len(text)
            body = text[start:end].strip()
            if body:
                chunks.append(body)
        return list(enumerate(_split_long_chunks(chunks, 1500)))

    # ④ 短文档 → 整篇 1 chunk
    if len(text) < 5000:
        return [(0, text.strip())]

    # ⑤ 兜底 → 按段落切
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged = _merge_paragraphs(paragraphs, target=800)
    return list(enumerate(merged))


def _split_long_chunks(chunks: list[str], max_len: int) -> list[str]:
    """超过 max_len 的 chunk 再按段落硬切。"""
    out: list[str] = []
    for c in chunks:
        if len(c) <= max_len:
            out.append(c)
            continue
        # 先按 \n\n 切
        paras = [p.strip() for p in c.split("\n\n") if p.strip()]
        merged = _merge_paragraphs(paras, target=max_len)
        out.extend(merged)
    return out


def _merge_paragraphs(paragraphs: list[str], target: int = 800) -> list[str]:
    """把零散段落合并到接近 target 字数。"""
    result: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= target:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            if buf:
                result.append(buf)
            buf = p
    if buf:
        result.append(buf)
    return result


# =========================================================================
# 2. 读取 per-file 元数据 → payload 补全
# =========================================================================
def _load_meta(doc_no: str) -> dict[str, Any]:
    """
    读单文件 meta JSON（标题/发布机关/印发日期）+ docs_catalog.json（效力级别）。
    两者合并后返回。
    """
    meta: dict[str, Any] = {}
    # 单文件 meta JSON
    for p in META_DIR.glob(f"{doc_no}_*.json"):
        if p.name == "docs_catalog.json":
            continue
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            break
        except Exception:
            pass
    # docs_catalog.json 补效力级别（catalog 的"编号"是字符串）
    if CATALOG_JSON.exists():
        try:
            catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
            for c in catalog:
                if str(c.get("编号", "")).zfill(2) == doc_no:
                    meta["效力级别"] = c.get("效力级别", "")
                    # 如果单文件 meta 没标题，从 catalog 取
                    if not meta.get("文件"):
                        meta["文件"] = c.get("标题", "")
                    break
        except Exception:
            pass
    return meta


# =========================================================================
# 3. Embedding（智谱 embedding-3，batch 防 429）
# =========================================================================
def _embed_batch(client, texts: list[str]) -> list[list[float]]:
    """分批调用智谱 embedding，返回原始向量列表。批次失败则拆单条重试。"""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        try:
            r = client.embeddings.create(model=EMBED_MODEL, input=batch)
            vec_map = {d.index: d.embedding for d in r.data}
            vectors.extend(vec_map[j] for j in range(len(batch)))
        except Exception as e:
            print(f"    batch FAIL ({len(batch)}条): {e} → 拆单条重试")
            for t in batch:
                vec = _embed_one_safe(client, t)
                vectors.append(vec)
    return vectors


def _embed_one_safe(client, text: str) -> list[float]:
    """单条 embedding，依次尝试：原文 → 截 2000 字 → 截 1000 字 → 零向量兜底。"""
    for truncate in [None, 2000, 1000, 500]:
        try:
            t = text[:truncate] if truncate else text
            r = client.embeddings.create(model=EMBED_MODEL, input=[t])
            return r.data[0].embedding
        except Exception:
            continue
    print(f"    所有截断方案均 FAIL, 补零向量 (原文 {len(text)} 字)")
    return [0.0] * EMBED_DIM


# =========================================================================
# MAIN
# =========================================================================
def main():
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct

    # —— key ——
    key = _read_zhipu_key()
    if not key:
        print("❌ 未找到 ZHIPU_API_KEY 或 ZHIPUAI_API_KEY")
        sys.exit(1)

    # —— Qdrant ——
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client_q = QdrantClient(path=str(QDRANT_PATH))

    # 已存在则删了重建
    if client_q.collection_exists(COLLECTION):
        client_q.delete_collection(COLLECTION)
    client_q.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    # —— 智谱 client ——
    from zhipuai import ZhipuAI
    client_z = ZhipuAI(api_key=key)

    # —— 遍历 13 篇文档 ——
    all_points: list[PointStruct] = []
    global_idx = 0
    files = sorted(CLEAN_DIR.glob("*.txt"))
    print(f"\n📂 发现 {len(files)} 篇文档")

    for txt_path in files:
        doc_no = _num(txt_path.name)
        meta = _load_meta(doc_no)
        title = meta.get("文件") or meta.get("标题") or txt_path.stem
        issuer = meta.get("发布机关") or ""
        issue_date = meta.get("印发日期") or ""
        level = meta.get("效力级别") or ""

        text = txt_path.read_text(encoding="utf-8-sig")
        chunks = split_into_chunks(text, doc_no)
        print(f"  [{doc_no}] {title[:20]:20s}  → {len(chunks):3d} chunks")

        if not chunks:
            continue

        # 拼接上下文头：用于 embedding 时让向量感知文档来源
        ctx_prefix = f"【文件：{title} | 发布机关：{issuer} | 发布日期：{issue_date}】\n"
        embed_texts = [
            _normalize_spaces(ctx_prefix + c) for _, c in chunks
        ]
        vectors = _embed_batch(client_z, embed_texts)

        for (chunk_idx, chunk_text), vec in zip(chunks, vectors):
            payload = {
                "doc_no": doc_no,
                "title": title,
                "issuer": issuer,
                "issue_date": issue_date,
                "level": level,
                "chunk_index": chunk_idx,
                "chunk_text": chunk_text[:2000],  # Qdrant payload 限长
                "snapshot": SNAPSHOT,
            }
            all_points.append(PointStruct(
                id=global_idx,
                vector=vec,
                payload=payload,
            ))
            global_idx += 1

    # —— 批量 upsert ——
    BATCH_UPSERT = 100
    for i in range(0, len(all_points), BATCH_UPSERT):
        client_q.upsert(
            collection_name=COLLECTION,
            points=all_points[i : i + BATCH_UPSERT],
        )

    count = client_q.count(COLLECTION).count
    print(f"\n✅ 向量化完成: 共 {count} chunks 入库 → {QDRANT_PATH}")

    # —— 抽样验证 ——
    sample = client_q.scroll(COLLECTION, limit=1, with_payload=True)[0][0]
    print(f"   抽样验证: id={sample.id}")
    print(f"     payload keys = {list(sample.payload.keys())}")
    print(f"     title = {sample.payload.get('title','')[:25]}")
    print(f"     issuer = {sample.payload.get('issuer','')[:15]}")
    print(f"     level = {sample.payload.get('level','')}")
    print(f"     snapshot = {sample.payload.get('snapshot','')}")

    # —— 显式释放（Windows portalocker 依赖）——
    client_q.close()


if __name__ == "__main__":
    main()
