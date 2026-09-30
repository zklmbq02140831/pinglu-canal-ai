import pathlib, json
base = pathlib.Path("data/processed")
clean, meta = base / "docs_clean", base / "docs_meta"
if not clean.exists() or not meta.exists():
    raise SystemExit(f"⚠ 目录不存在: {clean} 或 {meta}")

cat_path = meta / "docs_catalog.json"
cat = json.loads(cat_path.read_text(encoding="utf-8")) if cat_path.exists() else []
cat_map = {str(d.get("编号"))[:2]: d for d in cat}
jsons = {p.name[:2]: p for p in meta.glob("*.json") if p.name != "docs_catalog.json"}

missing, suspicious, nested = [], [], list(clean.glob("docs_meta"))
for f in sorted(clean.glob("*.txt")):
    raw = f.read_bytes()
    try: text = raw.decode("utf-8"); enc = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="replace"); f.write_text(text, encoding="utf-8"); enc = "gbk->utf-8"
    if "【正文开始】" in text[:200]: suspicious.append(f.name)
    info = json.loads(jsons[f.name[:2]].read_text(encoding="utf-8")) if f.name[:2] in jsons else {}
    if not info: missing.append(f.name)
    issuer = info.get("发布机关") or cat_map.get(f.name[:2], {}).get("发布机关", "-")
    level  = cat_map.get(f.name[:2], {}).get("效力级别", "-")
    print(f"{f.name[:2]} {len(text):>7}字 [{enc}] {issuer}｜{level}")

txt_ids = {f.name[:2] for f in clean.glob("*.txt")}
print(f"\ntxt:{len(txt_ids)}份 per-file JSON:{len(jsons)}份 catalog:{len(cat_map)}条")
if len(txt_ids) == 0 or len(jsons) == 0:
    raise SystemExit("⚠ 存在空目录——先确认文件到位,再谈对齐")
if nested: print(f"⚠ docs_clean里仍嵌套着{nested},删掉它")
print("三方对齐:", "✅一致" if txt_ids == set(jsons) == set(cat_map) else "⚠不一致")
print("缺JSON:", missing, " 疑似头部版:", suspicious or "无")
