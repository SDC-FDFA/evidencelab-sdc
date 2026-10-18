#!/usr/bin/env python3
"""Check (and optionally enrich) the metadata JSON files of a dataset folder.

Usage:
    python3 check_metadata.py data/sdc_evaluations/pdfs            # report only, changes nothing
    python3 check_metadata.py data/sdc_evaluations/pdfs --add-year # also adds "published_year" from
                                                                   # filenames like 2023_003.pdf
Run it before the pipeline: titles are required, and spelling differences in countries
create duplicate entries in the filter.
"""
import collections
import json
import pathlib
import re
import sys


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    add_year = "--add-year" in sys.argv
    if not args:
        print(__doc__)
        return 2
    folder = pathlib.Path(args[0])
    pdfs = {p.stem.lower(): p for p in folder.glob("*.pdf")} | {p.stem.lower(): p for p in folder.glob("*.PDF")}
    jsons = {p.stem.lower(): p for p in folder.glob("*.json")}
    problems, warnings = [], []
    countries, categories, languages = collections.Counter(), collections.Counter(), collections.Counter()
    titles = collections.defaultdict(list)
    changed = 0

    print(f"{len(pdfs)} PDFs, {len(jsons)} JSON files in {folder}\n")
    for stem in sorted(pdfs.keys() - jsons.keys()):
        problems.append(f"PDF without JSON: {pdfs[stem].name}")
    for stem in sorted(jsons.keys() - pdfs.keys()):
        problems.append(f"JSON without PDF: {jsons[stem].name}")

    for stem, path in sorted(jsons.items()):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path.name}: invalid JSON ({exc})")
            continue
        if not isinstance(data, dict):
            problems.append(f"{path.name}: top level must be an object")
            continue
        title = str(data.get("title") or "").strip()
        if not title:
            problems.append(f"{path.name}: missing title (the pipeline rejects documents without one)")
        titles[title.lower()].append(path.name)
        country = data.get("country")
        if not country:
            warnings.append(f"{path.name}: no country")
        elif isinstance(country, str):
            if "," in country and ";" not in country:
                warnings.append(f"{path.name}: country has a comma but no ';' -> {country!r} (use ';' between values)")
            for part in (c.strip() for c in country.split(";")):
                if part:
                    countries[part] += 1
        else:
            warnings.append(f"{path.name}: country is not text; use 'A; B' format")
        categories[str(data.get("category") or "(none)")] += 1
        languages[str(data.get("language") or "(none)")] += 1
        for key, value in data.items():
            if isinstance(value, str) and value != value.strip():
                warnings.append(f"{path.name}: {key!r} has leading/trailing spaces")
        if add_year and "published_year" not in data:
            match = re.match(r"^(\d{4})[_-]", path.stem)
            if match:
                data["published_year"] = match.group(1)
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                changed += 1
            else:
                warnings.append(f"{path.name}: cannot derive a year from the filename")

    for title, names in titles.items():
        if title and len(names) > 1:
            warnings.append(f"duplicate title in {', '.join(names)}")

    print("Countries / regions (check spelling variants):")
    for name, count in sorted(countries.items()):
        print(f"  {count:3}  {name}")
    print("\nCategories:", dict(categories))
    print("Languages :", dict(languages), "(config currently forces 'en' for everything)")
    if add_year:
        print(f"\nAdded published_year to {changed} file(s).")
    print(f"\nPROBLEMS ({len(problems)}) - fix before running:")
    print("\n".join(f"  - {p}" for p in problems) or "  none")
    print(f"\nWARNINGS ({len(warnings)}):")
    print("\n".join(f"  - {w}" for w in warnings) or "  none")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
