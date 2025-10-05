import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class ConvertResult:
    source_path: Path
    sheet_used: Optional[str]
    rows_out: int
    output_path: Optional[Path]
    error: Optional[str]


SUSPECT_SHEET_KEYWORDS = [
    "table 1",
    "top 100",
    "boys",
    "e&w",
    "england",
    "wales",
]


def normalize_string(value: object) -> str:
    if value is None:
        return ""
    s = str(value)
    return " ".join(s.replace("\n", " ").split()).strip()


def likely_sheet_names(sheet_names: List[str]) -> List[str]:
    # Rank sheets by heuristic score using keywords
    scored: List[Tuple[int, str]] = []
    for name in sheet_names:
        norm = normalize_string(name).lower()
        score = 0
        for kw in SUSPECT_SHEET_KEYWORDS:
            if kw in norm:
                score += 1
        if name.lower().strip() in ["table_1", "table 1", "table1"]:
            score += 3
        scored.append((score, name))
    # Highest score first; tie-breaker: keep order provided by Excel
    scored.sort(key=lambda t: t[0], reverse=True)
    ordered = [name for score, name in scored if score > 0]
    # Fallback: original ordering if nothing matched
    if not ordered:
        return list(sheet_names)
    return ordered


def detect_header_row(df_raw: pd.DataFrame) -> Optional[int]:
    # Search first 40 rows for a header that contains variants of Rank/Name/Count
    header_aliases = {
        "rank": {"rank", "position", "pos"},
        "name": {"name", "baby name", "boy name", "boys name", "baby boys name", "first name", "firstname", "given name", "label"},
        "count": {"count", "number", "frequency", "births", "occurrences"},
    }
    max_scan = min(40, len(df_raw))
    for i in range(max_scan):
        row = df_raw.iloc[i]
        texts = [normalize_string(v).lower() for v in row.tolist()]
        # Skip empty rows
        if not any(texts):
            continue
        has_rank = any(t in header_aliases["rank"] for t in texts)
        has_name = any(t in header_aliases["name"] for t in texts)
        has_count = any(t in header_aliases["count"] for t in texts)
        if (has_rank and has_name) or (has_name and has_count):
            return i
    return None


def build_dataframe_from_sheet(book_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    # Load sheet with no header to locate header row
    try:
        engine = None
        if book_path.suffix.lower() == ".xls":
            engine = "xlrd"
        df_raw = pd.read_excel(book_path, sheet_name=sheet_name, header=None, engine=engine)
    except Exception:
        return None

    header_idx = detect_header_row(df_raw)
    if header_idx is None:
        # Try with default header=0 as a last resort
        try:
            engine = None
            if book_path.suffix.lower() == ".xls":
                engine = "xlrd"
            df_guess = pd.read_excel(book_path, sheet_name=sheet_name, engine=engine)
            return df_guess
        except Exception:
            return None

    header_values = [normalize_string(v) for v in df_raw.iloc[header_idx].tolist()]
    df = df_raw.iloc[header_idx + 1 :].copy()
    df.columns = header_values
    return df


def select_columns(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    # Map columns by normalized name
    colmap: Dict[str, str] = {}
    for c in df.columns:
        key = normalize_string(c).lower()
        if not key:
            continue
        # De-duplicate by first occurrence
        if key not in colmap:
            colmap[key] = str(c)

    rank_aliases = ["rank", "position", "pos"]
    name_aliases = [
        "baby name",
        "boys name",
        "boy name",
        "name",
        "first name",
        "firstname",
        "given name",
        "label",
    ]
    count_aliases = ["count", "number", "frequency", "births", "occurrences"]

    def find_alias(aliases: List[str]) -> Optional[str]:
        for a in aliases:
            if a in colmap:
                return colmap[a]
        return None

    rank_col = find_alias(rank_aliases)
    name_col = find_alias(name_aliases)
    count_col = find_alias(count_aliases)

    # If missing rank, we can compute later; require at least name and count
    if name_col and count_col:
        return rank_col or "", name_col, count_col

    # Fallback: try first three non-empty columns as B,C,D or A,B,C patterns
    nonempty_cols = [c for c in df.columns if normalize_string(c)]
    if len(nonempty_cols) >= 3:
        c0, c1, c2 = nonempty_cols[:3]
        return str(c0), str(c1), str(c2)
    return None


def clean_and_rank(df: pd.DataFrame, cols: Tuple[str, str, str]) -> pd.DataFrame:
    rank_col, name_col, count_col = cols

    # Keep only columns of interest
    use_cols = [c for c in [rank_col, name_col, count_col] if c]
    df2 = df[use_cols].copy()

    # If selected columns produce DataFrames (duplicate column labels), coerce to Series
    def ensure_series(frame: pd.DataFrame, col: str, prefer_numeric: bool) -> pd.Series:
        obj = frame[col]
        if isinstance(obj, pd.DataFrame):
            # Choose a suitable single column by position to avoid duplicate-label recursion
            if prefer_numeric:
                for i in range(obj.shape[1]):
                    try:
                        series = pd.to_numeric(obj.iloc[:, i], errors="coerce")
                        if series.notna().sum() > 0:
                            return series
                    except Exception:
                        continue
                return obj.iloc[:, 0]
            else:
                for i in range(obj.shape[1]):
                    series = obj.iloc[:, i]
                    if series.dtype == object:
                        return series
                return obj.iloc[:, 0]
        return obj  # already a Series

    name_series = ensure_series(df2, name_col, prefer_numeric=False)
    count_series = ensure_series(df2, count_col, prefer_numeric=True)
    rank_series: Optional[pd.Series] = None
    if rank_col:
        rank_series = ensure_series(df2, rank_col, prefer_numeric=True)

    # Rebuild a normalized frame
    df2 = pd.DataFrame({
        name_col: name_series,
        count_col: count_series,
        **({rank_col: rank_series} if rank_series is not None else {}),
    })

    # Standardize names
    df2[name_col] = df2[name_col].map(normalize_string)
    df2[name_col] = df2[name_col].replace({"": pd.NA})

    # Coerce numerics
    if rank_col:
        df2[rank_col] = pd.to_numeric(df2[rank_col], errors="coerce")
    # Strip formatting and convert count
    count_str = df2[count_col].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    # Remove footnote markers like '*' or 'u' trailing
    count_str = count_str.str.replace(r"[^0-9\-]", "", regex=True)
    df2[count_col] = pd.to_numeric(count_str, errors="coerce")

    # Drop rows without name or without count
    df2 = df2.dropna(subset=[name_col, count_col])

    # Remove obvious header repeaters or footer text rows
    bad_name_markers = {
        "england and wales",
        "boys",
        "top 100",
        "table 1",
        "rank",
        "name",
        "count",
        "number",
    }
    df2 = df2[~df2[name_col].str.lower().isin(bad_name_markers)]

    # Collapse duplicates by name keeping max count
    df2 = df2.sort_values(by=[count_col], ascending=False)
    df2 = df2.groupby(name_col, as_index=False)[count_col].max()

    # Compute rank from count (ties share the same minimum rank)
    df2["Rank"] = df2[count_col].rank(method="min", ascending=False).astype(int)
    df2 = df2.sort_values(by=["Rank", name_col])

    # Final columns and names
    out = df2.rename(columns={name_col: "Baby name", count_col: "Count"})[["Rank", "Baby name", "Count"]]
    # Limit to 100 where applicable
    out = out[out["Rank"] <= 100]
    out = out.reset_index(drop=True)
    return out


def convert_workbook(path: Path) -> ConvertResult:
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        # Try specifying engine based on suffix
        try:
            engine = None
            if path.suffix.lower() == ".xls":
                engine = "xlrd"
            xls = pd.ExcelFile(path, engine=engine)
        except Exception as e2:
            return ConvertResult(source_path=path, sheet_used=None, rows_out=0, output_path=None, error=str(e2))

    candidate_sheets = likely_sheet_names(xls.sheet_names)

    last_error: Optional[str] = None
    for sheet in candidate_sheets:
        try:
            df = build_dataframe_from_sheet(path, sheet)
            if df is None or df.empty:
                last_error = "empty or unreadable sheet"
                continue
            cols = select_columns(df)
            if not cols:
                last_error = "cannot detect columns"
                continue
            out = clean_and_rank(df, cols)
            if out is None or out.empty:
                last_error = "no valid rows after cleaning"
                continue
            out_dir = path.parent / "csv"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{path.stem}_ew_boys.csv"
            out.to_csv(out_path, index=False)
            return ConvertResult(source_path=path, sheet_used=sheet, rows_out=len(out), output_path=out_path, error=None)
        except Exception as e:
            last_error = str(e)
            continue

    return ConvertResult(source_path=path, sheet_used=None, rows_out=0, output_path=None, error=last_error or "no suitable sheet found")


def run_batch(input_dir: Path) -> List[ConvertResult]:
    results: List[ConvertResult] = []
    for p in sorted(input_dir.glob("*.xls*")):
        if not p.is_file():
            continue
        res = convert_workbook(p)
        results.append(res)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch convert UK ONS boys Excel workbooks to CSV (E&W Top 100)")
    parser.add_argument("--input-dir", default="/Users/david/Names/Baby names/UK", help="Directory containing .xls/.xlsx files")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    results = run_batch(input_dir)
    ok = 0
    fail = 0
    for r in results:
        if r.error is None:
            ok += 1
            print(f"OK: {r.source_path.name} -> {r.output_path.name} (sheet='{r.sheet_used}', rows={r.rows_out})")
        else:
            fail += 1
            print(f"FAIL: {r.source_path.name} :: {r.error}")

    print(f"\nCompleted. Files processed: {len(results)} | Success: {ok} | Failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


