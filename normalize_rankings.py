import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path("/Users/david/Names/Baby names")
OUTPUT_JSON = Path("/Users/david/Names/normalized_rankings.json")
OUTPUT_PARQUET_DIR = Path("/Users/david/Names/normalized_rankings_parquet")


def normalize_nsw() -> List[Dict]:
    path = ROOT / "NSW" / "popular-baby-names-1952-to-2024.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    # Try to identify columns
    cmap = {c.lower().strip(): c for c in df.columns}
    name_col = cmap.get("name") or cmap.get("first name") or list(df.columns)[0]
    year_col = cmap.get("year")
    rank_col = cmap.get("rank") or cmap.get("ranking")
    count_col = cmap.get("count") or cmap.get("number") or cmap.get("births")
    sex_col = cmap.get("sex") or cmap.get("gender")
    if sex_col in df.columns:
        df = df[df[sex_col].astype(str).str.upper().isin(["M", "MALE", "BOY", "BOYS"])].copy()
    out = []
    for _, r in df.iterrows():
        name = str(r[name_col]).strip()
        if not name:
            continue
        yr = pd.to_numeric(r.get(year_col), errors="coerce") if year_col else None
        yr = int(yr) if pd.notna(yr) else None
        rk = pd.to_numeric(r.get(rank_col), errors="coerce") if rank_col else None
        rk = int(rk) if pd.notna(rk) else None
        ct = pd.to_numeric(r.get(count_col), errors="coerce") if count_col else None
        ct = int(ct) if pd.notna(ct) else None
        out.append({"region": "NSW", "name": name.title(), "year": yr, "rank": rk, "count": ct})
    return out


def normalize_uk() -> List[Dict]:
    # Prefer combined CSV if available
    csv_all = ROOT / "UK" / "UK_all_years_boys_EW.csv"
    records: List[Dict] = []
    if csv_all.exists():
        df = pd.read_csv(csv_all)
        cmap = {c.lower().strip(): c for c in df.columns}
        # Try a robust detection
        year_col = cmap.get("year") or cmap.get("yr")
        rank_col = cmap.get("rank") or cmap.get("ranking") or cmap.get("position")
        count_col = cmap.get("count") or cmap.get("number") or cmap.get("births") or cmap.get("frequency")
        # Detect name column heuristically: prefer columns named like "name" else the most stringy column
        candidates = [c for c in df.columns if df[c].dtype == object]
        name_col = cmap.get("name") or cmap.get("first name") or cmap.get("baby name")
        if name_col is None and candidates:
            def stringiness(col: str) -> float:
                s = df[col].astype(str)
                alpha = s.str.contains(r"[A-Za-z]", regex=True, na=False).mean()
                digits = s.str.contains(r"\d", regex=True, na=False).mean()
                return alpha - digits
            name_col = max(candidates, key=stringiness)
        for _, r in df.iterrows():
            name = str(r[name_col]).strip()
            if not name:
                continue
            yr = pd.to_numeric(r.get(year_col), errors="coerce") if year_col else None
            yr = int(yr) if pd.notna(yr) else None
            rk = pd.to_numeric(r.get(rank_col), errors="coerce") if rank_col else None
            rk = int(rk) if pd.notna(rk) else None
            ct = pd.to_numeric(r.get(count_col), errors="coerce") if count_col else None
            ct = int(ct) if pd.notna(ct) else None
            records.append({"region": "UK", "name": name.title(), "year": yr, "rank": rk, "count": ct})
        return records
    # Fallback: parse individual XLS/XLSX via pandas
    uk_dir = ROOT / "UK"
    for p in uk_dir.glob("*.xls*"):
        try:
            df = pd.read_excel(p)
            cmap = {c.lower().strip(): c for c in df.columns}
            name_col = cmap.get("name") or cmap.get("first name") or list(df.columns)[0]
            year_col = cmap.get("year")
            rank_col = cmap.get("rank") or cmap.get("ranking")
            count_col = cmap.get("count") or cmap.get("number") or cmap.get("births")
            # Infer year from filename if not present in sheet
            year_hint: Optional[int] = None
            for token in p.stem.replace("_", " ").replace("-", " ").split():
                if token.isdigit() and len(token) == 4:
                    year_hint = int(token)
                    break
            for _, r in df.iterrows():
                name = str(r[name_col]).strip()
                if not name:
                    continue
                yr = pd.to_numeric(r.get(year_col), errors="coerce") if year_col else year_hint
                yr = int(yr) if yr is not None and pd.notna(yr) else year_hint
                rk = pd.to_numeric(r.get(rank_col), errors="coerce") if rank_col else None
                rk = int(rk) if pd.notna(rk) else None
                ct = pd.to_numeric(r.get(count_col), errors="coerce") if count_col else None
                ct = int(ct) if pd.notna(ct) else None
                records.append({"region": "UK", "name": name.title(), "year": yr, "rank": rk, "count": ct})
        except Exception:
            continue
    return records


def normalize_vic() -> List[Dict]:
    records: List[Dict] = []
    vic_dir = ROOT / "Victoria"
    for p in vic_dir.glob("*.xls*"):
        try:
            xls = pd.ExcelFile(p)
            # choose sheets that look like boys/males if available, otherwise all
            candidate_sheets = [s for s in xls.sheet_names if any(k in s.lower() for k in ["boy", "male"]) ] or xls.sheet_names
            year_hint: Optional[int] = None
            for token in p.stem.replace("_", " ").replace("-", " ").split():
                if token.isdigit() and len(token) == 4:
                    year_hint = int(token)
                    break
            for sheet in candidate_sheets:
                df = xls.parse(sheet)
                cmap = {c.lower().strip(): c for c in df.columns}
                # detect name column heuristically
                name_col = cmap.get("name") or cmap.get("first name") or cmap.get("given name")
                if name_col is None:
                    candidates = [c for c in df.columns if df[c].dtype == object]
                    if candidates:
                        def stringiness(col: str) -> float:
                            s = df[col].astype(str)
                            alpha = s.str.contains(r"[A-Za-z]", regex=True, na=False).mean()
                            digits = s.str.contains(r"\d", regex=True, na=False).mean()
                            return alpha - digits
                        name_col = max(candidates, key=stringiness)
                year_col = cmap.get("year")
                rank_col = cmap.get("rank") or cmap.get("ranking")
                count_col = cmap.get("count") or cmap.get("number") or cmap.get("births") or cmap.get("frequency")
                gender_col = cmap.get("sex") or cmap.get("gender")
                df_use = df
                if gender_col in df.columns:
                    df_use = df[df[gender_col].astype(str).str.upper().isin(["M", "MALE", "BOY", "BOYS"])].copy()
                if rank_col is None and count_col in df_use.columns and name_col in df_use.columns:
                    try:
                        df_use[count_col] = pd.to_numeric(df_use[count_col], errors="coerce")
                        df_use = df_use.sort_values(by=[count_col, name_col], ascending=[False, True])
                        df_use["__rank"] = range(1, len(df_use) + 1)
                        rank_col = "__rank"
                    except Exception:
                        pass
                for _, r in df_use.iterrows():
                    name = str(r[name_col]).strip()
                    if not name:
                        continue
                    yr = pd.to_numeric(r.get(year_col), errors="coerce") if year_col else year_hint
                    yr = int(yr) if yr is not None and pd.notna(yr) else year_hint
                    rk = pd.to_numeric(r.get(rank_col), errors="coerce") if rank_col else None
                    rk = int(rk) if rk is not None and pd.notna(rk) else None
                    ct = pd.to_numeric(r.get(count_col), errors="coerce") if count_col else None
                    ct = int(ct) if ct is not None and pd.notna(ct) else None
                    records.append({"region": "VIC", "name": name.title(), "year": yr, "rank": rk, "count": ct})
        except Exception:
            continue
    return records


def normalize_usa() -> List[Dict]:
    records: List[Dict] = []
    usa_dir = ROOT / "USA"
    # Prefer pre-filtered boys CSVs if present
    for p in usa_dir.glob("yob*_boys.csv"):
        try:
            df = pd.read_csv(p)
            cmap = {c.lower().strip(): c for c in df.columns}
            name_col = cmap.get("name") or list(df.columns)[0]
            count_col = cmap.get("count") or cmap.get("number")
            year_hint: Optional[int] = None
            try:
                year_hint = int(p.stem.split("_")[0].replace("yob", ""))
            except Exception:
                pass
            for _, r in df.iterrows():
                name = str(r[name_col]).strip()
                if not name:
                    continue
                ct = pd.to_numeric(r.get(count_col), errors="coerce") if count_col else None
                ct = int(ct) if pd.notna(ct) else None
                records.append({"region": "USA", "name": name.title(), "year": year_hint, "rank": None, "count": ct})
        except Exception:
            continue
    # Fallback to SSA text files: name,sex,count
    for p in usa_dir.glob("yob*.txt"):
        try:
            df = pd.read_csv(p, header=None, names=["name", "sex", "count"])
            df = df[df["sex"].astype(str).str.upper() == "M"].copy()
            year_hint: Optional[int] = None
            try:
                year_hint = int(p.stem.replace("yob", ""))
            except Exception:
                pass
            for _, r in df.iterrows():
                name = str(r["name"]).strip()
                if not name:
                    continue
                ct = pd.to_numeric(r.get("count"), errors="coerce")
                ct = int(ct) if pd.notna(ct) else None
                records.append({"region": "USA", "name": name.title(), "year": year_hint, "rank": None, "count": ct})
        except Exception:
            continue
    return records


def main():
    all_records: List[Dict] = []
    all_records.extend(normalize_nsw())
    all_records.extend(normalize_vic())
    all_records.extend(normalize_uk())
    all_records.extend(normalize_usa())

    # Build name -> region -> list of {year, rank, count}
    name_map: Dict[str, Dict[str, List[Dict]]] = {}
    for rec in all_records:
        name = rec["name"]
        region = rec["region"]
        entry = {k: rec[k] for k in ["year", "rank", "count"]}
        name_map.setdefault(name, {}).setdefault(region, []).append(entry)

    # Sort each list by year desc
    for region_map in name_map.values():
        for region, items in region_map.items():
            items.sort(key=lambda x: (x["year"] if x["year"] is not None else -1), reverse=True)

    OUTPUT_JSON.write_text(json.dumps(name_map, indent=2))
    print(f"Wrote {OUTPUT_JSON} with {len(name_map)} names")

    # Also write partitioned Parquet for lazy loading in the app
    try:
        df = pd.DataFrame(all_records)  # columns: region, name, year, rank, count
        if not df.empty:
            def _initial(s: str) -> str:
                if not s:
                    return "#"
                ch = s[0].upper()
                return ch if ("A" <= ch <= "Z") else "#"
            df["initial"] = df["name"].astype(str).map(_initial)
            OUTPUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
            written_files = 0
            for ini, g in df.groupby("initial"):
                outp = OUTPUT_PARQUET_DIR / f"{ini}.parquet"
                g.drop(columns=["initial"], errors="ignore").to_parquet(outp, index=False)
                written_files += 1
            print(f"Wrote Parquet partitions to {OUTPUT_PARQUET_DIR} ({written_files} files)")
    except Exception as e:
        print(f"Warning: failed to write Parquet partitions: {e}")


if __name__ == "__main__":
    main()


