from pathlib import Path
import argparse
import sys
import pandas as pd


def convert_yob(input_path: Path, output_path: Path) -> int:
    # SSA format: name,sex,count (no header)
    try:
        df = pd.read_csv(input_path, header=None, names=["name", "sex", "count"], dtype={0: str, 1: str, 2: int})
    except Exception:
        # Some files may have counts with commas; handle generically
        df = pd.read_csv(input_path, header=None, names=["name", "sex", "count"], dtype=str)
        df["count"] = (
            df["count"].astype(str).str.replace(",", "", regex=False).str.strip()
        )
        df["count"] = pd.to_numeric(df["count"], errors="coerce")

    # Filter to male only
    df = df[df["sex"].astype(str).str.upper() == "M"].copy()
    df = df.dropna(subset=["name", "count"])  
    # Rank by count desc; ties share rank (min). SSA files typically are already sorted, but we recompute to be safe
    df["Rank"] = df["count"].rank(method="min", ascending=False).astype(int)
    df = df.sort_values(["Rank", "name"])  

    out = df.rename(columns={"name": "name", "count": "count"})[["Rank", "name", "count"]]
    out.columns = ["rank", "name", "count"]
    out.to_csv(output_path, index=False)
    return len(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert SSA yobYYYY.txt to CSV with rank,name,count (boys only)")
    parser.add_argument("--input", default="/Users/david/Names/Baby names/USA/yob2024.txt")
    parser.add_argument("--output", default="/Users/david/Names/Baby names/USA/yob2024_boys.csv")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = convert_yob(input_path, output_path)
    print(f"Wrote {rows} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


