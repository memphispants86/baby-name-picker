import math
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import streamlit as st
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

APP_TITLE = "Baby A name picker"
SURNAME_MARTIN_PROPORTION_DEFAULT = 0.00224  # ≈ 0.224% of Australians have surname Martin
NORMALIZED_JSON_PATH = Path("/Users/david/Names/normalized_rankings.json")  # kept for local fallback
LOG_PATH = Path("/Users/david/Names/app.log")
PARQUET_DIR = Path("/Users/david/Names/normalized_rankings_parquet")
### Expected births parameters (requested)
# Annual totals used for scaling
AUS_TOTAL_BIRTHS = 304_000
VIC_TOTAL_BIRTHS = 72_906
USA_TOTAL_BIRTHS = 3_784_000
INTERNATIONAL_TOTAL_BIRTHS = 4_772_000

# Configure logging (file-based)
_logger = logging.getLogger("baby_name_picker")
if not _logger.handlers:
    _logger.setLevel(logging.DEBUG)
    try:
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        _logger.addHandler(fh)
    except Exception:
        # If file handler fails (permissions), fall back to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        _logger.addHandler(ch)


# --------------------------------------------------------------------------------------
# Database setup
# --------------------------------------------------------------------------------------

Base = declarative_base()


class Spouse(Base):
    __tablename__ = "spouses"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)

    ratings = relationship("Rating", back_populates="spouse")


class Name(Base):
    __tablename__ = "names"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)

    ratings = relationship("Rating", back_populates="name_obj")
    rankings = relationship("RankingEntry", back_populates="name_obj")


class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True)
    name_id = Column(Integer, ForeignKey("names.id"), nullable=False, index=True)
    spouse_id = Column(Integer, ForeignKey("spouses.id"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)  # 0-10 integer
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    name_obj = relationship("Name", back_populates="ratings")
    spouse = relationship("Spouse", back_populates="ratings")

    __table_args__ = (
        UniqueConstraint("name_id", "spouse_id", name="uq_rating_per_spouse_per_name"),
    )


class RankingSource(Base):
    __tablename__ = "ranking_sources"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)  # e.g., "NSW 2024", "Australia 2023"
    year = Column(Integer, nullable=True)
    notes = Column(String, nullable=True)

    entries = relationship("RankingEntry", back_populates="source_obj")


class RankingEntry(Base):
    __tablename__ = "ranking_entries"

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("ranking_sources.id"), nullable=False, index=True)
    name_id = Column(Integer, ForeignKey("names.id"), nullable=False, index=True)
    rank = Column(Integer, nullable=True)
    count = Column(Integer, nullable=True)  # births count (per-source/year), if available

    source_obj = relationship("RankingSource", back_populates="entries")
    name_obj = relationship("Name", back_populates="rankings")

    __table_args__ = (
        UniqueConstraint("source_id", "name_id", name="uq_source_name"),
    )


def get_engine_and_session() -> Tuple:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        engine = create_engine(database_url)
    else:
        db_path = Path(__file__).resolve().parent / "names.db"
        engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
    SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
    return engine, SessionLocal


@st.cache_resource(show_spinner=False)
def get_session_factory():
    engine, SessionLocal = get_engine_and_session()
    Base.metadata.create_all(engine)
    return SessionLocal


def bootstrap_data_if_needed(session):
    # Ensure spouses exist
    for spouse_name in ["Nancy", "David"]:
        existing = session.execute(select(Spouse).where(Spouse.name == spouse_name)).scalar_one_or_none()
        if existing is None:
            session.add(Spouse(name=spouse_name))

    # Load initial names from List.txt if names table empty
    existing_name = session.execute(select(func.count(Name.id))).scalar_one()
    if existing_name == 0:
        candidates: List[Path] = [
            Path(__file__).resolve().parent / "List.txt",
            Path("/Users/david/Names/List.txt"),
        ]
        list_path: Optional[Path] = None
        for c in candidates:
            if c.exists():
                list_path = c
                break
        if list_path is not None:
            with list_path.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            names = [nm for nm in lines if nm]
            for nm in names:
                session.add(Name(name=nm))
    session.commit()
    


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def get_or_create_spouse(session, spouse_name: str) -> Spouse:
    spouse = session.execute(select(Spouse).where(Spouse.name == spouse_name)).scalar_one_or_none()
    if spouse is None:
        spouse = Spouse(name=spouse_name)
        session.add(spouse)
        session.commit()
        session.refresh(spouse)
    return spouse


def get_next_unrated_name(session, spouse_id: int) -> Optional[Name]:
    rated_name_ids = [rid for (rid,) in session.execute(
        select(Rating.name_id).where(Rating.spouse_id == spouse_id)
    ).all()]
    q = select(Name).where(~Name.id.in_(rated_name_ids)).order_by(Name.name.asc())
    return session.execute(q).scalars().first()


def set_rating(session, name_id: int, spouse_id: int, rating_value: int):
    existing = session.execute(
        select(Rating).where(Rating.name_id == name_id, Rating.spouse_id == spouse_id)
    ).scalar_one_or_none()
    if existing is None:
        existing = Rating(name_id=name_id, spouse_id=spouse_id, rating=rating_value, updated_at=datetime.utcnow())
        session.add(existing)
    else:
        existing.rating = rating_value
        existing.updated_at = datetime.utcnow()
    session.commit()


def clear_rating(session, name_id: int, spouse_id: int):
    existing = session.execute(
        select(Rating).where(Rating.name_id == name_id, Rating.spouse_id == spouse_id)
    ).scalar_one_or_none()
    if existing is not None:
        session.delete(existing)
        session.commit()


def fetch_all_names(session) -> List[Name]:
    return session.execute(select(Name).order_by(Name.name.asc())).scalars().all()


def fetch_ratings_for_spouse(session, spouse_id: int) -> Dict[int, int]:
    rows = session.execute(select(Rating.name_id, Rating.rating).where(Rating.spouse_id == spouse_id)).all()
    return {name_id: rating for name_id, rating in rows}


def compute_percentile_weights(values: List[int]) -> Dict[int, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Use rank-based percentile with mid-rank handling for ties
    value_to_weight: Dict[int, float] = {}
    for v in sorted(set(sorted_vals)):
        lower = sum(1 for x in sorted_vals if x < v)
        equal = sum(1 for x in sorted_vals if x == v)
        percentile = (lower + 0.5 * equal) / n  # 0..1
        value_to_weight[v] = percentile * 10.0  # scale to 0..10
    return value_to_weight


def compute_zscore_weights(values: List[int]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean_v = float(sum(values)) / float(len(values))
    # Use population variance; guard for degenerate cases
    var = sum((float(v) - mean_v) ** 2 for v in values) / float(len(values))
    std = math.sqrt(var) if var > 0 else 1.0
    if not math.isfinite(mean_v) or not math.isfinite(std) or std <= 0:
        return (mean_v if math.isfinite(mean_v) else 0.0), 1.0
    return mean_v, std


def get_ranking_summary_for_name(session, name_id: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    # Returns (best_rank, most_recent_year_with_count, best_count_for_that_year)
    entries = session.execute(select(RankingEntry).where(RankingEntry.name_id == name_id)).scalars().all()
    if not entries:
        return None, None, None
    best_rank: Optional[int] = None
    for e in entries:
        if e.rank is not None:
            best_rank = e.rank if best_rank is None else min(best_rank, e.rank)

    # pick the most recent year with at least one non-null count
    year_to_counts: Dict[int, List[int]] = {}
    for e in entries:
        if e.count is not None and e.source_obj and e.source_obj.year is not None:
            year_to_counts.setdefault(e.source_obj.year, []).append(e.count)
    if not year_to_counts:
        return best_rank, None, None
    most_recent_year = max(year_to_counts.keys())
    # Use max to avoid double counting if multiple sources cover same population
    best_count = max(year_to_counts[most_recent_year]) if year_to_counts[most_recent_year] else None
    return best_rank, most_recent_year, best_count


def _region_from_source_label(label: str) -> Optional[str]:
    norm = label.strip().lower()
    if norm.startswith("nsw"):
        return "NSW"
    if norm.startswith("victoria") or norm.startswith("vic"):
        return "VIC"
    if norm.startswith("uk") or "ons" in norm:
        return "UK"
    if norm.startswith("usa") or "ssa" in norm:
        return "USA"
    return None


def get_normalized_json_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "normalized_rankings.json",
        NORMALIZED_JSON_PATH,
    ]
    for p in candidates:
        if p.exists():
            return p
    # default to repo-local path
    return candidates[0]


@st.cache_data(show_spinner=False)
def load_normalized_map(mtime: float) -> Optional[Dict[str, Dict[str, List[Dict]]]]:
    try:
        # Prefer lazy parquet index if present for memory efficiency
        if PARQUET_DIR.exists():
            # Return a lightweight index: available initials
            initials = sorted([p.stem for p in PARQUET_DIR.glob("*.parquet")])
            return {"__parquet__": {"initials": initials}}
        path = get_normalized_json_path()
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_region_summaries_for_name_by_text(session, name_text: str) -> Dict[str, str]:
    # Prefer normalized JSON if present for speed and consistency
    json_path = get_normalized_json_path()
    mtime = json_path.stat().st_mtime if json_path.exists() else 0.0
    name_map = load_normalized_map(mtime)
    if name_map is not None:
        # If using parquet partitions, load only the needed partition
        if isinstance(name_map, dict) and "__parquet__" in name_map and PARQUET_DIR.exists():
            try:
                ini = (name_text[:1].upper() if name_text else "#")
                part = PARQUET_DIR / f"{ini}.parquet"
                if part.exists():
                    df_part = pd.read_parquet(part)
                    df_part = df_part[df_part["name"].str.lower() == name_text.lower()]
                    region_map: Dict[str, List[Dict]] = {}
                    for _, r in df_part.iterrows():
                        region_map.setdefault(r["region"], []).append({
                            "year": int(r["year"]) if pd.notna(r["year"]) else None,
                            "rank": int(r["rank"]) if pd.notna(r["rank"]) else None,
                            "count": int(r["count"]) if pd.notna(r["count"]) else None,
                        })
                    # Build summaries from region_map
                    summaries: Dict[str, str] = {}
                    for region in ["NSW", "VIC", "UK", "USA"]:
                        items = region_map.get(region, [])
                        if not items:
                            summaries[region] = "-"
                            continue
                        items_sorted = sorted(items, key=lambda x: (x.get("year") or -1), reverse=True)
                        parts: List[str] = []
                        for it in items_sorted:
                            year = it.get("year")
                            rank = it.get("rank")
                            count = it.get("count")
                            year_str = str(year) if year is not None else "?"
                            if rank is not None and count is not None:
                                parts.append(f"{year_str}: {rank} ({count})")
                            elif rank is not None:
                                parts.append(f"{year_str}: {rank}")
                            elif count is not None:
                                parts.append(f"{year_str}: - ({count})")
                            else:
                                parts.append(f"{year_str}:")
                        summaries[region] = "; ".join(parts)
                    return summaries
            except Exception as e:
                _logger.exception(f"parquet lookup failed for name={name_text}: {e}")
        # Look up by exact case first; fallback to case-insensitive match
        region_map = name_map.get(name_text)
        if region_map is None:
            for k in name_map.keys():
                if k.lower() == name_text.lower():
                    region_map = name_map[k]
                    break
        summaries: Dict[str, str] = {}
        for region in ["NSW", "VIC", "UK", "USA"]:
            items = region_map.get(region, []) if region_map else []
            if not items:
                summaries[region] = "-"
                continue
            parts: List[str] = []
            for it in items:
                year = it.get("year")
                rank = it.get("rank")
                count = it.get("count")
                year_str = str(year) if year is not None else "?"
                if rank is not None and count is not None:
                    parts.append(f"{year_str}: {rank} ({count})")
                elif rank is not None:
                    parts.append(f"{year_str}: {rank}")
                elif count is not None:
                    parts.append(f"{year_str}: - ({count})")
                else:
                    parts.append(f"{year_str}:")
            summaries[region] = "; ".join(parts)
        return summaries

    # Fallback to DB entries if JSON missing
    rows = session.execute(select(RankingEntry, RankingSource).where(RankingEntry.name_id == name_id).join(RankingSource, RankingEntry.source_id == RankingSource.id)).all()
    region_to_items: Dict[str, List[Tuple[int, Optional[int], Optional[int]]]] = {}
    for entry, src in rows:
        region = _region_from_source_label(src.source or "") or ""
        if not region:
            continue
        year_val = src.year if src.year is not None else None
        region_to_items.setdefault(region, []).append((year_val if year_val is not None else -1, entry.rank, entry.count))

    summaries: Dict[str, str] = {}
    for region in ["NSW", "VIC", "UK", "USA"]:
        items = region_to_items.get(region, [])
        if not items:
            summaries[region] = "-"
            continue
        items_sorted = sorted(items, key=lambda t: (t[0] if t[0] is not None else -1), reverse=True)
        parts: List[str] = []
        for year, rank, count in items_sorted:
            year_str = str(year) if (year is not None and year >= 0) else "?"
            if rank is not None and count is not None:
                parts.append(f"{year_str}: {rank} ({count})")
            elif rank is not None:
                parts.append(f"{year_str}: {rank}")
            elif count is not None:
                parts.append(f"{year_str}: - ({count})")
            else:
                parts.append(f"{year_str}:")
        summaries[region] = "; ".join(parts)
    return summaries


@st.cache_data(show_spinner=False)
def build_results_dataframe(method: str, data_version: int) -> pd.DataFrame:
    _logger.debug(f"build_results_dataframe method={method} dv={data_version}")
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        # Only compute for names that have at least one rating
        all_names = fetch_all_names(session)
        ratings_present: Dict[int, bool] = {}
        spouses = session.execute(select(Spouse)).scalars().all()
        for s in spouses:
            rmap = fetch_ratings_for_spouse(session, s.id)
            for nid in rmap.keys():
                ratings_present[nid] = True
        filtered_names = [nm for nm in all_names if ratings_present.get(nm.id)]
        _logger.debug(f"names_total={len(all_names)} names_rated={len(filtered_names)}")

        spouse_names = [s.name for s in spouses]
        # Ensure default spouses ordering
        ordered_spouses = [sn for sn in ["Nancy", "David"] if sn in spouse_names] + [sn for sn in spouse_names if sn not in ["Nancy", "David"]]

        # Ratings per spouse
        ratings_by_spouse: Dict[str, Dict[int, int]] = {}
        for s in spouses:
            ratings_by_spouse[s.name] = fetch_ratings_for_spouse(session, s.id)

        # Weight maps per spouse
        percentile_maps: Dict[str, Dict[int, float]] = {}
        zscore_params: Dict[str, Tuple[float, float]] = {}
        for s in spouses:
            values = list(ratings_by_spouse[s.name].values())
            percentile_maps[s.name] = compute_percentile_weights(values)
            try:
                zscore_params[s.name] = compute_zscore_weights(values)
            except Exception as e:
                _logger.exception(f"zscore_params error spouse={s.name} values={values}")
                zscore_params[s.name] = (0.0, 1.0)

        records: List[Dict] = []
        for nm in all_names:
            row: Dict = {"Name": nm.name}
            weighted_values: List[float] = []
            for sp_name in ordered_spouses:
                r = ratings_by_spouse.get(sp_name, {}).get(nm.id)
                row[f"{sp_name} rating"] = r
                if r is not None:
                    try:
                        if method == "Percentile":
                            w = percentile_maps[sp_name].get(r)
                            # If not enough data or missing, fall back to raw
                            w = float(r) if w is None else float(w)
                        else:  # Z-score
                            mean_v, std_v = zscore_params[sp_name]
                            if std_v == 0 or not math.isfinite(std_v):
                                w = float(r)
                            else:
                                # Map mean to 5, 1 std dev = 2 points; clip 0..10
                                w = 5.0 + ((float(r) - mean_v) / std_v) * 2.0
                                w = max(0.0, min(10.0, w))
                    except Exception as e:
                        _logger.exception(f"weighting error method={method} spouse={sp_name} r={r}")
                        w = float(r)
                    row[f"{sp_name} weighted"] = round(w, 3)
                    weighted_values.append(w)
                else:
                    row[f"{sp_name} weighted"] = None

            # Combined score: average of available spouse weighted scores
            if weighted_values:
                row["Overall score"] = round(sum(weighted_values) / float(len(weighted_values)), 3)
            else:
                row["Overall score"] = None

            # Region summaries
            region_summaries = get_region_summaries_for_name_by_text(session, nm.name)
            row["NSW"] = region_summaries.get("NSW", "-")
            row["VIC"] = region_summaries.get("VIC", "-")
            row["UK"] = region_summaries.get("UK", "-")
            row["USA"] = region_summaries.get("USA", "-")

            # Expected births (Aus) and (International) per user specification
            expected_aus = 0.0
            expected_intl = 0.0
            json_path3 = get_normalized_json_path()
            mtime2 = json_path3.stat().st_mtime if json_path3.exists() else 0.0
            name_map = load_normalized_map(mtime2)
            def _latest_count(items: List[Dict]) -> Optional[int]:
                if not items:
                    return None
                try:
                    items_sorted = sorted(items, key=lambda x: (x.get("year") or -1), reverse=True)
                except Exception:
                    items_sorted = items
                for it in items_sorted:
                    cnt = it.get("count")
                    if cnt is None:
                        continue
                    try:
                        cval = int(cnt)
                        return cval
                    except Exception:
                        try:
                            cval = int(float(cnt))
                            return cval
                        except Exception:
                            continue
                return None

            vic_cnt: Optional[int] = None
            nsw_cnt: Optional[int] = None
            usa_cnt: Optional[int] = None
            if name_map is not None:
                region_map = name_map.get(nm.name) or next((name_map[k] for k in name_map.keys() if k.lower() == nm.name.lower()), None)
                if region_map:
                    vic_cnt = _latest_count(region_map.get("VIC", []))
                    nsw_cnt = _latest_count(region_map.get("NSW", []))
                    usa_cnt = _latest_count(region_map.get("USA", []))

            # Expected births (Aus)
            if vic_cnt is not None and VIC_TOTAL_BIRTHS > 0:
                expected_aus = (float(vic_cnt) / float(VIC_TOTAL_BIRTHS)) * float(AUS_TOTAL_BIRTHS)
            elif nsw_cnt is not None and VIC_TOTAL_BIRTHS > 0:
                expected_aus = (float(nsw_cnt) / float(VIC_TOTAL_BIRTHS)) * float(AUS_TOTAL_BIRTHS)
            elif usa_cnt is not None and USA_TOTAL_BIRTHS > 0:
                expected_aus = (float(usa_cnt) / float(USA_TOTAL_BIRTHS)) * float(AUS_TOTAL_BIRTHS)
            else:
                expected_aus = 0.0

            # Expected births (International)
            if usa_cnt is not None and USA_TOTAL_BIRTHS > 0:
                expected_intl = (float(usa_cnt) / float(USA_TOTAL_BIRTHS)) * float(INTERNATIONAL_TOTAL_BIRTHS)
            else:
                expected_intl = 0.0

            _logger.debug(
                f"expected_births name={nm.name} VIC={vic_cnt} NSW={nsw_cnt} USA={usa_cnt} "
                f"Aus={expected_aus:.3f} Intl={expected_intl:.3f}"
            )

            # Round to whole births for display
            row["Expected births (Aus)"] = int(round(expected_aus))
            row["Expected births (International)"] = int(round(expected_intl))

            # Martin surname expectation (≈0.224%)
            martin_factor = float(SURNAME_MARTIN_PROPORTION_DEFAULT)
            row["Expected Martin births (Aus)"] = int(round(expected_aus * martin_factor))
            row["Expected Martin births (International)"] = int(round(expected_intl * martin_factor))

            records.append(row)

        df = pd.DataFrame.from_records(records)
        if not df.empty:
            # Sort by overall score desc; if needed, could include hidden best_rank
            df = df.sort_values(by=["Overall score"], ascending=[False], na_position="last")
        _logger.debug(f"results rows={len(df)}")
        return df
    finally:
        session.close()


def _read_table_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path)
    if suffix in [".tsv"]:
        return pd.read_csv(path, sep="\t")
    # Try CSV with automatic delimiter detection
    try:
        return pd.read_csv(path)
    except Exception:
        # Fallback to semicolon
        return pd.read_csv(path, sep=";")


def _detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    colmap = {c.lower().strip(): c for c in df.columns}
    def find(*aliases: List[str]) -> Optional[str]:
        for alias_group in aliases:
            for alias in alias_group:
                if alias in colmap:
                    return colmap[alias]
        return None

    name_col = find(["name"], ["first_name"], ["firstname"], ["given name"], ["baby name"], ["boy name"], ["girl name"], ["label"])  # type: ignore
    rank_col = find(["rank"], ["ranking"], ["position"], ["pos"])  # optional
    count_col = find(["count"], ["births"], ["frequency"], ["number"], ["n"])  # optional
    year_col = find(["year"], ["yr"])  # optional
    gender_col = find(["sex"], ["gender"])  # optional
    return {"name": name_col, "rank": rank_col, "count": count_col, "year": year_col, "gender": gender_col}


def import_rankings_from_dataframe(session, df: pd.DataFrame, add_missing_names: bool, source_label: str, year_hint: Optional[int], notes: Optional[str]) -> int:
    cols = _detect_columns(df)
    name_col = cols["name"]
    if name_col is None:
        raise ValueError("No name column found")
    rank_col = cols["rank"]
    count_col = cols["count"]
    year_col = cols["year"]
    gender_col = cols["gender"]

    # If a gender column exists, prefer rows that are not aggregate; keep all otherwise
    df_use = df.copy()
    if gender_col is not None:
        # keep all; caller may have pre-filtered
        pass

    # Determine years in data
    years: List[Optional[int]]
    if year_hint is not None and (year_col is None or df_use[year_col].nunique() == 0):
        years = [year_hint]
    elif year_col is not None:
        years_raw = pd.to_numeric(df_use[year_col], errors="coerce").dropna().astype(int).tolist()
        years = sorted(list(set(years_raw)))
        if not years and year_hint is not None:
            years = [year_hint]
    else:
        years = [year_hint] if year_hint is not None else [None]

    def insert_block(block: pd.DataFrame, file_year: Optional[int]) -> int:
        # Create/find source per (source_label, file_year)
        source_obj = session.execute(select(RankingSource).where(
            RankingSource.source == source_label, RankingSource.year == file_year
        )).scalar_one_or_none()
        if source_obj is None:
            source_obj = RankingSource(source=source_label, year=file_year, notes=notes)
            session.add(source_obj)
            session.commit()
            session.refresh(source_obj)

        # If rank column missing, derive rank by descending count within block
        local = block.copy()
        if rank_col is None and count_col is not None and count_col in local.columns:
            try:
                local[count_col] = pd.to_numeric(local[count_col], errors="coerce")
                local = local.sort_values(by=[count_col, name_col], ascending=[False, True])
                local["__rank"] = range(1, len(local) + 1)
                effective_rank_col = "__rank"
            except Exception:
                effective_rank_col = None
        else:
            effective_rank_col = rank_col

        inserted_local = 0
        for _, row in local.iterrows():
            nm_raw = str(row[name_col]).strip()
            if not nm_raw:
                continue
            name_obj = session.execute(select(Name).where(Name.name == nm_raw)).scalar_one_or_none()
            if name_obj is None:
                if add_missing_names:
                    name_obj = Name(name=nm_raw)
                    session.add(name_obj)
                    session.flush()
                else:
                    continue

            rank_val: Optional[int] = None
            if effective_rank_col is not None and pd.notna(row.get(effective_rank_col)):
                try:
                    rank_val = int(row[effective_rank_col])
                except Exception:
                    rank_val = None

            count_val: Optional[int] = None
            if count_col is not None and pd.notna(row.get(count_col)):
                try:
                    count_val = int(row[count_col])
                except Exception:
                    count_val = None

            entry = session.execute(select(RankingEntry).where(
                RankingEntry.source_id == source_obj.id,
                RankingEntry.name_id == name_obj.id,
            )).scalar_one_or_none()
            if entry is None:
                session.add(RankingEntry(source_id=source_obj.id, name_id=name_obj.id, rank=rank_val, count=count_val))
                inserted_local += 1
            else:
                if rank_val is not None:
                    entry.rank = rank_val
                if count_val is not None:
                    entry.count = count_val
        session.commit()
        return inserted_local

    total_inserted = 0
    if year_col is not None and len(years) > 1:
        for yr in years:
            block = df_use[pd.to_numeric(df_use[year_col], errors="coerce").astype('Int64') == int(yr)]
            total_inserted += insert_block(block, int(yr))
    else:
        total_inserted += insert_block(df_use, years[0] if years else None)

    return total_inserted


def import_rankings_from_file(session, path: Path, add_missing_names: bool) -> int:
    try:
        df = _read_table_any(path)
    except Exception:
        return 0
    # Derive source label and year hint from path
    source_label = f"{path.parent.name} / {path.name}"
    year_hint: Optional[int] = None
    for token in path.stem.replace("_", " ").replace("-", " ").split():
        if token.isdigit() and len(token) == 4:
            try:
                year = int(token)
                if 1900 < year < 2100:
                    year_hint = year
                    break
            except Exception:
                pass
    return import_rankings_from_dataframe(session, df, add_missing_names, source_label, year_hint, notes=None)


def import_rankings_from_folder(session, root: Path, add_missing_names: bool) -> Dict[str, int]:
    stats = {"files": 0, "inserted": 0}
    if not root.exists():
        return stats
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"]:
            inserted = import_rankings_from_file(session, p, add_missing_names)
            stats["files"] += 1
            stats["inserted"] += int(inserted)
    return stats


def auto_ingest_rankings(session):
    root = Path("/Users/david/Names/Baby names")
    # NSW combined CSV (boys and girls) with columns similar to Name, Gender, Year, Count, Rank
    nsw_csv = root / "NSW" / "popular-baby-names-1952-to-2024.csv"
    if nsw_csv.exists():
        try:
            df = pd.read_csv(nsw_csv)
            # Keep only boys by default for this app; adjust if needed
            gender_col = None
            for c in df.columns:
                if str(c).lower().strip() in ["sex", "gender"]:
                    gender_col = c
                    break
            df_use = df
            if gender_col is not None:
                df_use = df[df[gender_col].astype(str).str.upper().isin(["M", "MALE", "BOY", "BOYS"])].copy()
            import_rankings_from_dataframe(session, df_use, add_missing_names=False, source_label="NSW", year_hint=None, notes="NSW 1952-2024")
        except Exception:
            pass

    # Victoria: many XLSX files with top 100; read each
    vic_dir = root / "Victoria"
    if vic_dir.exists():
        for p in vic_dir.glob("*.xls*"):
            try:
                import_rankings_from_file(session, p, add_missing_names=False)
            except Exception:
                pass

    # UK: ONS boys files per year
    uk_dir = root / "UK"
    if uk_dir.exists():
        csv_all = uk_dir / "UK_all_years_boys_EW.csv"
        if csv_all.exists():
            try:
                df = pd.read_csv(csv_all)
                # Expect columns like Name, Year, Rank, Count
                import_rankings_from_dataframe(session, df, add_missing_names=False, source_label="UK ONS", year_hint=None, notes="UK England & Wales boys all years")
            except Exception:
                pass
        else:
            for p in uk_dir.glob("*.xls*"):
                try:
                    import_rankings_from_file(session, p, add_missing_names=False)
                except Exception:
                    pass

    # USA: SSA yobYYYY.txt files in format name,sex,count. Keep male rows
    usa_dir = root / "USA"
    if usa_dir.exists():
        # Prefer any pre-filtered boys CSV for latest year if present
        for p in usa_dir.glob("yob*_boys.csv"):
            try:
                year_hint = None
                try:
                    year_hint = int(p.stem.split("_")[0].replace("yob", ""))
                except Exception:
                    pass
                df = pd.read_csv(p)
                import_rankings_from_dataframe(session, df, add_missing_names=False, source_label="USA SSA", year_hint=year_hint, notes=p.name)
            except Exception:
                pass
        for p in usa_dir.glob("yob*.txt"):
            try:
                df = pd.read_csv(p, header=None, names=["name", "sex", "count"])
                df = df[df["sex"].astype(str).str.upper() == "M"].copy()
                year_hint = None
                try:
                    year_hint = int(p.stem.replace("yob", ""))
                except Exception:
                    pass
                import_rankings_from_dataframe(session, df, add_missing_names=False, source_label="USA SSA", year_hint=year_hint, notes=str(p.name))
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Version token to bust cache when ratings/names change
if "data_version" not in st.session_state:
    st.session_state["data_version"] = 0

SessionFactory = get_session_factory()
session = SessionFactory()
bootstrap_data_if_needed(session)

with st.sidebar:
    st.header("Settings")
    spouse_selection = st.selectbox("Who's rating?", options=["Nancy", "David"], index=0)

tabs = st.tabs(["Rate", "Results", "Add names"])  # keep concise


# --------------------------------------------------------------------------------------
# Tab: Rate
# --------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Rate names 0–10")
    spouse = get_or_create_spouse(session, spouse_selection)

    # Choose a name
    all_names = fetch_all_names(session)
    ratings_map = fetch_ratings_for_spouse(session, spouse.id)

    # Option to focus on unrated names
    unrated_only = st.checkbox("Show only unrated names", value=True)
    if unrated_only:
        name_options = [n for n in all_names if n.id not in ratings_map]
    else:
        name_options = all_names

    next_unrated = get_next_unrated_name(session, spouse.id)
    default_index = 0
    if next_unrated is not None and next_unrated in name_options:
        default_index = name_options.index(next_unrated)

    if not name_options:
        st.success("No names left to rate in this view.")
    else:
        selected_name = st.selectbox(
            "Select a name",
            options=name_options,
            index=default_index,
            format_func=lambda n: n.name,
            key=f"select_name_{spouse.id}",
        )
        existing_rating = ratings_map.get(selected_name.id)

        with st.form(key=f"rating_form_{spouse.id}"):
            rating_value = st.slider("Rating", min_value=0, max_value=10, value=int(existing_rating) if existing_rating is not None else 5)
            cols = st.columns([1, 1, 4])
            save_clicked = cols[0].form_submit_button("Save")
            clear_clicked = cols[1].form_submit_button("Clear")
            if save_clicked:
                set_rating(session, selected_name.id, spouse.id, int(rating_value))
                st.success(f"Saved: {selected_name.name} = {int(rating_value)} for {spouse.name}")
                st.session_state["data_version"] += 1
                st.rerun()
            if clear_clicked:
                clear_rating(session, selected_name.id, spouse.id)
                st.info(f"Cleared rating for {selected_name.name} ({spouse.name})")
                st.session_state["data_version"] += 1
                st.rerun()

    # Explicit editor for existing ratings
    rated_names = [n for n in all_names if n.id in ratings_map]
    if rated_names:
        st.divider()
        st.subheader("Edit existing rating")
        edit_name = st.selectbox(
            "Select a rated name",
            options=rated_names,
            format_func=lambda n: n.name,
            key=f"edit_select_name_{spouse.id}",
        )
        current = ratings_map.get(edit_name.id, 5)
        with st.form(key=f"edit_rating_form_{spouse.id}"):
            new_val = st.slider("New rating", min_value=0, max_value=10, value=int(current))
            cols2 = st.columns([1, 1, 4])
            save2 = cols2[0].form_submit_button("Update")
            clear2 = cols2[1].form_submit_button("Clear")
            if save2:
                set_rating(session, edit_name.id, spouse.id, int(new_val))
                st.success(f"Updated: {edit_name.name} = {int(new_val)} for {spouse.name}")
                st.session_state["data_version"] += 1
                st.rerun()
            if clear2:
                clear_rating(session, edit_name.id, spouse.id)
                st.info(f"Cleared rating for {edit_name.name} ({spouse.name})")
                st.session_state["data_version"] += 1
                st.rerun()


# --------------------------------------------------------------------------------------
# Tab: Results
# --------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Results and combined ranking")
    # Auto-ingest external rankings once on first visit to Results if database has none
    try:
        rank_count = session.execute(select(func.count(RankingEntry.id))).scalar_one()
    except Exception:
        rank_count = 0
    if rank_count == 0 and not st.session_state.get("ingested_auto", False):
        with st.status("Loading external rankings from 'Baby names'…", expanded=True) as status:
            try:
                auto_ingest_rankings(session)
                status.update(label="External rankings loaded", state="complete", expanded=False)
            except Exception as e:
                st.warning(f"Auto-ingest failed: {e}")
        st.session_state["ingested_auto"] = True
        st.session_state["data_version"] += 1
        build_results_dataframe.clear()
        st.rerun()
    weighting_method = st.radio(
        "Weighting method",
        options=["Percentile", "Z-score"],
        index=0,
        help="Percentile maps each spouse's ratings to their own distribution; Z-score normalizes by mean and spread.",
        horizontal=True,
    )
    st.caption("Debug: switch methods safely; guards against zero-variance and empty sets.")
    df = build_results_dataframe(weighting_method, st.session_state.get("data_version", 0))
    if df.empty:
        # Provide a hint if normalized JSON is present but names may not match
        if get_normalized_json_path().exists():
            st.info("No ratings yet. Start on the Rate tab. Regional columns will appear once you add ratings.")
        else:
            st.info("No ratings yet. Start on the Rate tab.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="baby_name_results.csv", mime="text/csv")


    


# --------------------------------------------------------------------------------------
# Tab: Add names
# --------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Add new names")
    st.caption("Enter one name per line. Case is preserved.")
    text = st.text_area("Names", height=150)
    if st.button("Add names"):
        raw = [ln.strip() for ln in text.splitlines()]
        to_add = [nm for nm in raw if nm]
        if not to_add:
            st.info("No names to add.")
        else:
            added = 0
            for nm in to_add:
                exists = session.execute(select(Name).where(Name.name == nm)).scalar_one_or_none()
                if exists is None:
                    session.add(Name(name=nm))
                    added += 1
            session.commit()
            st.success(f"Added {added} names.")
            st.session_state["data_version"] += 1
            build_results_dataframe.clear()
            st.rerun()


# Close session at end of script execution
session.close()


