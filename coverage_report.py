"""Per-document check after an ingestion: status, chunk counts and section types.

Run (from the project folder, local or on the server with the `dc` alias instead of `docker compose`):
    docker compose exec -T -e DS=sdc_evaluations api python - < coverage_report.py
DS is the dataset's data_subdir (collections documents_<DS> and chunks_<DS>).
Documents are listed from fewest to most chunks; suspicious ones are flagged.
"""
import collections
import os

from qdrant_client import QdrantClient

c = QdrantClient(url="http://qdrant:6333", api_key=os.getenv("QDRANT_API_KEY"))
ds = os.environ["DS"]

docs, offset = [], None
while True:
    pts, offset = c.scroll(f"documents_{ds}", limit=200, offset=offset,
                           with_payload=["map_title", "sys_status"], with_vectors=False)
    docs += pts
    if offset is None:
        break
print("document status:", dict(collections.Counter(str(d.payload.get("sys_status")) for d in docs)))

per_doc = collections.defaultdict(collections.Counter)
offset = None
while True:
    pts, offset = c.scroll(f"chunks_{ds}", limit=500, offset=offset,
                           with_payload=["doc_id", "tag_section_type"], with_vectors=False)
    for p in pts:
        per_doc[str(p.payload.get("doc_id"))][str(p.payload.get("tag_section_type"))] += 1
    if offset is None:
        break

total = sum(sum(v.values()) for v in per_doc.values())
print(f"{len(docs)} documents, {total} chunks\n")
rows = []
for d in docs:
    sections = per_doc.get(str(d.id), collections.Counter())
    n = sum(sections.values())
    top = sections.most_common(1)[0] if sections else ("-", 0)
    flag = ""
    if n < 20:
        flag = "  <-- FEW CHUNKS (scanned PDF? parse failed?)"
    elif top[1] / n > 0.8:
        flag = "  <-- ONE SECTION TYPE DOMINATES"
    elif "None" in sections:
        flag = "  <-- UNTAGGED CHUNKS"
    rows.append((n, f"{n:5} chunks  top={top[0]}({top[1]})  {str(d.payload.get('map_title'))[:60]}{flag}"))
for _, line in sorted(rows):
    print(line)
