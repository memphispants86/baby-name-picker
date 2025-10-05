from pathlib import Path
import re
import sys
import pandas as pd


def extract_year_from_filename(name: str) -> int:
    # Prefer a standalone 4-digit year in the filename, else 0
    tokens = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", name)
    if tokens:
        try:
            y = int(tokens[0])
            if 1900 < y < 2100:
                return y
        except Exception:
            pass
    return 0


def main() -> int:
    uk_dir = Path("/Users/david/Names/Baby names/UK")
    csv_dir = uk_dir / "csv"
    if not csv_dir.exists():
        print(f"No csv directory found: {csv_dir}", file=sys.stderr)
        return 2

    records = []
    for p in sorted(csv_dir.glob("*_ew_boys.csv")):
        try:
            df = pd.read_csv(p)
            year = extract_year_from_filename(p.stem)
            if year == 0:
                # Try to read from file stem tokens
                year = extract_year_from_filename(p.name)
            df.insert(0, "Year", year)
            df.insert(1, "Source file", p.name)
            records.append(df)
        except Exception as e:
            print(f"WARN: failed reading {p.name}: {e}")

    if not records:
        print("No CSVs found to aggregate.")
        return 1

    all_df = pd.concat(records, ignore_index=True)

    # Ensure types
    all_df["Year"] = pd.to_numeric(all_df["Year"], errors="coerce").fillna(0).astype(int)
    all_df["Rank"] = pd.to_numeric(all_df["Rank"], errors="coerce")
    all_df["Count"] = pd.to_numeric(all_df["Count"], errors="coerce")

    # Sort by Year asc, then Rank asc
    all_df = all_df.sort_values(by=["Year", "Rank"], ascending=[True, True])

    out_path = uk_dir / "UK_all_years_boys_EW.csv"
    all_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} rows={len(all_df)} from {len(records)} files")
    # Quick sanity: show min/max year and sample top/bottom rows
    if len(all_df) > 0:
        years = [y for y in all_df["Year"].unique().tolist() if int(y) > 0]
        if years:
            print(f"Years: {min(years)}–{max(years)}")
        # Show small preview
        print(all_df.head(3).to_string(index=False))
        print("…")
        print(all_df.tail(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


