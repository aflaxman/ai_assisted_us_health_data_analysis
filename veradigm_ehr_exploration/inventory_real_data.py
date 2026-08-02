#!/usr/bin/env python
"""Profile a real delivery and emit aggregate, non-identifiable metadata.

This is the only script in the project that touches limited-use data, and the
analyst runs it inside the limited-use environment. Its output is a JSON profile
-- counts, fill rates, distributions, code-set frequencies -- intended to be
reviewed and then carried back out to calibrate ``simulate_vnehr.py``.

Disclosure controls, in order of strictness:

* Columns whose name ends in ``_id`` never have their values listed, only
  counts, a distinct-count estimate, and a numeric min/max.
* Every listed categorical or free-text value must occur at least ``--min-cell``
  times; rarer values are counted and reported only as a suppressed total.
* Free-text columns emit at most ``--top-n`` values, subject to the same floor.

Nothing is hardcoded about the delivery location: ``--root`` is required.

    python inventory_real_data.py --root <delivery-root> --out outputs/real_data_profile.json
    python inventory_real_data.py --root <delivery-root> --sample-rows 5000000 --min-cell 50
"""

import argparse
import bz2
import csv
import gzip
import io
import json
import lzma
import os
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import roles
import schema_config

try:  # optional fast path; the limited-use environment may not have it
    import duckdb
except ImportError:
    duckdb = None

TABULAR_EXTENSIONS = {".csv", ".psv", ".txt", ".tsv", ".tab", ".dat", ".parquet", ".pqt"}
COMPRESSION_EXTENSIONS = {".gz": "gzip", ".bz2": "bzip2", ".zip": "zip", ".zst": "zstd",
                          ".xz": "xz", ".sz": "snappy"}
DELIMITERS = {"|": "pipe", ",": "comma", "\t": "tab", ";": "semicolon"}

# Shard suffixes seen in deliveries: Name_001, Name-part-00003, Name.2024.
SHARD_SUFFIX = re.compile(
    r"(?:[._-](?:part|chunk|shard|file|split)?[._-]?\d{1,6})+$|"
    r"(?:[._-]\d{4}[-_]?\d{0,2}[-_]?\d{0,2})$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# banners
# --------------------------------------------------------------------------

def banner(lines):
    width = min(max(len(x) for x in lines) + 4, 100)
    print("=" * width, file=sys.stderr)
    for line in lines:
        print(f"  {line}", file=sys.stderr)
    print("=" * width, file=sys.stderr, flush=True)


def start_banner(args):
    banner([
        "REAL-DATA INVENTORY -- AGGREGATE STATISTICS ONLY",
        f"root            : {args.root}",
        f"output          : {args.out}",
        f"suppression     : values occurring < {args.min_cell} times are withheld",
        f"free-text top-N : {args.top_n}",
        f"row cap / table : {args.sample_rows:,}" if args.sample_rows else "row cap / table : none",
        "id columns       : values NEVER emitted (name ends in _id)",
        "This script writes no row-level data. Review the JSON before",
        "moving it out of the limited-use environment.",
    ])


def end_banner(args, path, n_tables, n_errors):
    banner([
        "INVENTORY COMPLETE -- AGGREGATE STATISTICS ONLY",
        f"profile written : {path}",
        f"tables profiled : {n_tables}",
        f"errors recorded : {n_errors}",
        f"suppression     : min cell size {args.min_cell}; id values never listed",
        "REVIEW THE JSON BEFORE MOVING IT OUT OF THE LIMITED-USE ENVIRONMENT.",
    ])


def progress(message):
    print(f"  {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


# --------------------------------------------------------------------------
# file walk, compression and format detection
# --------------------------------------------------------------------------

def walk_delivery(root, include_underscore_dirs=False):
    """Every file under root with size, extension and inferred compression."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        skipped_dir = (not include_underscore_dirs
                       and any(part.startswith((".", "_")) for part in rel_dir.split(os.sep)
                               if part not in (".", "")))
        dirnames.sort()
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = None
            stem, compression = strip_compression(filename)
            ext = os.path.splitext(stem)[1].lower()
            entries.append({
                "path": os.path.relpath(full, root),
                "name": filename,
                "size_bytes": size,
                "extension": ext or None,
                "compression": compression,
                "tabular": ext in TABULAR_EXTENSIONS,
                "skipped": bool(skipped_dir),
                "skip_reason": "metadata directory (leading '.' or '_')" if skipped_dir else None,
            })
    return entries


def strip_compression(filename):
    """('SomeTable.psv.gz') -> ('SomeTable.psv', 'gzip')."""
    stem, ext = os.path.splitext(filename)
    if ext.lower() in COMPRESSION_EXTENSIONS:
        return stem, COMPRESSION_EXTENSIONS[ext.lower()]
    return filename, None


def open_text(path, compression, encoding="utf-8"):
    """Text handle for a possibly compressed delivery file."""
    if compression == "gzip":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding=encoding, errors="replace")
    if compression == "bzip2":
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding=encoding, errors="replace")
    if compression == "xz":
        return io.TextIOWrapper(lzma.open(path, "rb"), encoding=encoding, errors="replace")
    if compression == "zip":
        archive = zipfile.ZipFile(path)
        member = archive.namelist()[0]
        return io.TextIOWrapper(archive.open(member), encoding=encoding, errors="replace")
    return open(path, "r", encoding=encoding, errors="replace", newline="")


def sniff_format(path, compression, sample_lines=40):
    """Delimiter, header presence and quoting, inferred from the first lines."""
    out = {"delimiter": None, "delimiter_name": None, "has_header": None,
           "quotechar": None, "sample_error": None}
    try:
        with open_text(path, compression) as handle:
            head = [handle.readline() for _ in range(sample_lines)]
    except Exception as exc:  # noqa: BLE001 - keep going on any read failure
        out["sample_error"] = f"{type(exc).__name__}: {exc}"
        return out
    head = [line for line in head if line]
    if not head:
        out["sample_error"] = "empty file"
        return out

    counts = {d: min(line.count(d) for line in head[:5]) for d in DELIMITERS}
    delimiter = max(counts, key=counts.get)
    if counts[delimiter] == 0:
        delimiter = None
    out["delimiter"] = delimiter
    out["delimiter_name"] = DELIMITERS.get(delimiter)
    out["quotechar"] = '"' if any('"' in line for line in head[:5]) else None

    if delimiter:
        first = head[0].rstrip("\r\n").split(delimiter)
        numericish = sum(bool(re.fullmatch(r"-?\d+(\.\d+)?", v.strip().strip('"'))) for v in first)
        out["has_header"] = numericish <= max(1, len(first) // 10)
        out["n_columns_first_row"] = len(first)
        try:
            dialect = csv.Sniffer().sniff("".join(head[:20]), delimiters="".join(DELIMITERS))
            out["quotechar"] = dialect.quotechar
        except csv.Error:
            pass
    return out


def match_tables(entries, cfg):
    """Group delivery files under the documented table each one appears to be.

    Matching is deliberately conservative: the file stem, with shard suffixes
    removed, must equal a table name or contain it as a whole token. Anything
    else is reported as unmatched rather than guessed at.
    """
    names = [cfg.table(role) for role in cfg.table_roles()]
    lookup = {normalize(name): name for name in names}
    # Longest name first, so a compound name is never claimed by its prefix.
    token_lookup = dict(sorted(
        ((tuple(name.lower().split("_")), name) for name in names),
        key=lambda item: -len(item[0])))
    groups, unmatched = {}, []
    for entry in entries:
        if entry["skipped"] or not entry["tabular"]:
            continue
        stem = os.path.splitext(strip_compression(entry["name"])[0])[0]
        stem = SHARD_SUFFIX.sub("", stem)
        key = normalize(stem)
        table = lookup.get(key)
        if table is None:
            tokens = [t for t in re.split(r"[^A-Za-z0-9]+", stem.lower()) if t]
            for name_tokens, name in token_lookup.items():
                width = len(name_tokens)
                if any(tuple(tokens[i:i + width]) == name_tokens
                       for i in range(len(tokens) - width + 1)):
                    table = name
                    break
        if table is None:
            unmatched.append(entry["path"])
            continue
        groups.setdefault(table, []).append(entry)
    return groups, unmatched


# --------------------------------------------------------------------------
# per-column accumulators
# --------------------------------------------------------------------------

RESERVOIR = 200_000        # values kept for quantiles / length percentiles
TRACK_DISTINCT = 60_000    # exact value counts kept before falling back to KMV
KMV_K = 4096               # k-minimum-values sketch size for distinct estimates

NULL_LITERALS = {"", "na", "n/a", "null", "nan", "none", "\\n", ".", "-", "unknown_null"}
DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}"),
    re.compile(r"^\d{2}/\d{2}/\d{4}"),
    re.compile(r"^\d{8}$"),
)


def is_id_column(name):
    """Id columns never have their values listed, whatever else they look like."""
    lowered = name.lower()
    return lowered.endswith("_id") or lowered in {"id", "npi", "mrn"}


def code_columns(cfg):
    """Column names holding a vocabulary or geographic code, for this delivery.

    Such codes parse as numbers but are categories. Summarising them as quantiles
    both fabricates codes that do not exist (the interpolated ones) and emits real
    rare codes without passing the min-cell floor, so they are routed to the
    suppressed value-count path instead.
    """
    return {name.lower() for name in cfg.field_names(roles.CODE_FIELDS)}


class Reservoir:
    """Fixed-size uniform sample, for quantiles without holding the column."""

    def __init__(self, size, rng):
        self.size = size
        self.rng = rng
        self.seen = 0
        self.values = np.empty(0, dtype="float64")

    def add(self, values):
        values = np.asarray(values, dtype="float64")
        values = values[np.isfinite(values)]
        if not values.size:
            return
        if self.values.size < self.size:
            take = min(self.size - self.values.size, values.size)
            self.values = np.concatenate([self.values, values[:take]])
            self.seen += take
            values = values[take:]
            if not values.size:
                return
        # Replace with probability size/seen, vectorised over the incoming block.
        positions = self.seen + np.arange(values.size)
        keep = self.rng.random(values.size) < (self.size / (positions + 1.0))
        slots = self.rng.integers(0, self.size, size=int(keep.sum()))
        self.values[slots] = values[keep]
        self.seen += values.size

    def quantiles(self, qs):
        if not self.values.size:
            return None
        return [round(float(v), 4) for v in np.quantile(self.values, qs)]


class KMV:
    """K-minimum-values distinct-count sketch; stdlib hashing, no dependencies."""

    def __init__(self, k=KMV_K):
        self.k = k
        self.mins = np.full(k, np.inf)

    def add(self, values):
        if not len(values):
            return
        hashed = pd.util.hash_array(np.asarray(values, dtype=object)).astype("float64")
        hashed = hashed / float(2 ** 64)
        merged = np.concatenate([self.mins, hashed])
        merged.sort()
        self.mins = merged[:self.k]

    def estimate(self):
        finite = self.mins[np.isfinite(self.mins)]
        if finite.size < self.k:
            return int(finite.size)
        largest = finite[-1]
        if largest <= 0:
            return int(finite.size)
        return int(round((self.k - 1) / largest))


class ColumnStats:
    """Streaming statistics for one column of one table."""

    def __init__(self, name, field, rng, max_distinct, codes):
        self.name = name
        self.is_id = is_id_column(name)
        self.is_code = name.lower() in codes
        self.declared = field["type"]["raw"] if field else None
        self.declared_base = field["type"]["base"] if field else None
        self.max_distinct = max_distinct
        self.total = 0
        self.non_null = 0
        self.counter = Counter()
        self.counter_overflow = False
        self.kmv = KMV()
        self.numeric_ok = 0
        self.numeric_bad = 0
        self.n_num = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = None
        self.max = None
        self.reservoir = Reservoir(RESERVOIR, rng)
        self.date_ok = 0
        self.date_bad = 0
        self.date_min = None
        self.date_max = None
        self.year_counts = Counter()
        self.length_reservoir = Reservoir(RESERVOIR, rng)
        self.length_max = 0

    def update(self, block):
        """Accumulate one chunk of a column, given as an object-dtype Series."""
        self.total += len(block)
        text = block.astype("string")
        text = text.mask(text.str.strip().str.lower().isin(NULL_LITERALS))
        present = text.dropna()
        self.non_null += len(present)
        if not len(present):
            return

        values = present.to_numpy(dtype=object)
        self.kmv.add(values)
        if not self.counter_overflow:
            self.counter.update(values)
            if len(self.counter) > TRACK_DISTINCT:
                self.counter_overflow = True
                self.counter = Counter(dict(self.counter.most_common(self.max_distinct * 4)))

        lengths = present.str.len().to_numpy(dtype="float64")
        self.length_reservoir.add(lengths)
        self.length_max = max(self.length_max, int(np.nanmax(lengths)))

        numeric = pd.to_numeric(present, errors="coerce")
        good = numeric.dropna()
        self.numeric_ok += len(good)
        self.numeric_bad += len(present) - len(good)
        if len(good):
            array = good.to_numpy(dtype="float64")
            self.n_num += array.size
            self.sum += float(array.sum())
            self.sumsq += float((array ** 2).sum())
            low, high = float(array.min()), float(array.max())
            self.min = low if self.min is None else min(self.min, low)
            self.max = high if self.max is None else max(self.max, high)
            self.reservoir.add(array)

        if self._date_like(present):
            parsed = pd.to_datetime(present, errors="coerce", format="mixed")
            good = parsed.dropna()
            self.date_ok += len(good)
            self.date_bad += len(present) - len(good)
            if len(good):
                low, high = good.min(), good.max()
                self.date_min = low if self.date_min is None else min(self.date_min, low)
                self.date_max = high if self.date_max is None else max(self.date_max, high)
                self.year_counts.update(good.dt.year.astype("int64").tolist())

    def _date_like(self, present):
        if self.declared_base in ("date", "datetime") or self.name.lower().endswith("_date"):
            return True
        sample = present.head(50)
        hits = sum(any(p.match(v) for p in DATE_PATTERNS) for v in sample)
        return hits >= max(1, int(0.8 * len(sample)))

    # ---- reporting -------------------------------------------------------

    def inferred_dtype(self):
        if self.date_ok and self.date_ok >= 0.9 * self.non_null:
            return "date"
        if self.numeric_ok and self.numeric_ok >= 0.95 * max(self.non_null, 1):
            return "numeric"
        return "text"

    def distinct_estimate(self):
        if not self.counter_overflow:
            return {"exact": len(self.counter)}
        return {"estimate": self.kmv.estimate(), "method": "k-minimum-values", "capped": True}

    def value_report(self, min_cell, limit=None):
        """Value counts, with everything below the disclosure floor suppressed.

        ``limit`` caps how many values are listed; low-cardinality categoricals
        pass None so the whole (already suppressed) value set is reported.
        """
        if self.is_id:
            return None, {"note": "id column: values never emitted"}
        items = self.counter.most_common()
        kept = [(v, c) for v, c in items if c >= min_cell]
        low_frequency_rows = sum(c for v, c in items if c < min_cell)
        listed = kept if limit is None else kept[:limit]
        return (
            [{"value": str(v)[:200], "count": int(c)} for v, c in listed],
            {"values_below_min_cell": len(items) - len(kept),
             "rows_in_suppressed_values": int(low_frequency_rows),
             "values_beyond_limit": max(0, len(kept) - len(listed)),
             "value_counts_are_capped": self.counter_overflow},
        )

    def to_dict(self, min_cell, top_n, max_distinct):
        out = {
            "declared_type": self.declared,
            "inferred_dtype": self.inferred_dtype(),
            "rows_seen": self.total,
            "non_null": self.non_null,
            "null_fraction": round(1 - self.non_null / self.total, 6) if self.total else None,
            "distinct": self.distinct_estimate(),
            "is_id_column": self.is_id,
            "is_code_column": self.is_code,
        }
        dtype = out["inferred_dtype"]
        n_distinct = self.distinct_estimate().get("exact")

        if self.n_num and not self.is_code:
            mean = self.sum / self.n_num
            variance = max(self.sumsq / self.n_num - mean ** 2, 0.0)
            out["numeric"] = {
                "count": self.n_num,
                "mean": round(mean, 4),
                "std": round(variance ** 0.5, 4),
                "min": round(self.min, 4),
                "max": round(self.max, 4),
                "deciles": self.reservoir.quantiles(np.arange(0.1, 1.0, 0.1)),
                "parse_failures": self.numeric_bad,
            }
        if self.date_ok:
            out["dates"] = {
                "min": self.date_min.strftime("%Y-%m-%d") if self.date_min is not None else None,
                "max": self.date_max.strftime("%Y-%m-%d") if self.date_max is not None else None,
                "count_by_year": {str(y): int(c) for y, c in sorted(self.year_counts.items())},
                "parse_failures": self.date_bad,
            }
        if self.is_id:
            out["disclosure"] = "id column: no values emitted"
            if self.n_num:
                out["numeric"] = {k: v for k, v in out["numeric"].items()
                                  if k in ("count", "min", "max")}
            out.pop("dates", None)
            return out

        low_cardinality = n_distinct is not None and n_distinct <= max_distinct
        if low_cardinality and dtype != "date":
            values, notes = self.value_report(min_cell)
            out["categorical"] = {"values": values, "suppression": notes}
        elif self.is_code:
            # High-cardinality vocabularies (NDC, RxNorm) still get a suppressed
            # top-N; without this branch they would report no values at all,
            # because the free-text branch below only fires on non-numeric data.
            values, notes = self.value_report(min_cell, top_n)
            out["codes"] = {"top_values": values, "suppression": notes}
        elif dtype == "text":
            values, notes = self.value_report(min_cell, top_n)
            out["free_text"] = {
                "length": {
                    "max": self.length_max,
                    "percentiles": self.length_reservoir.quantiles([0.05, 0.25, 0.5, 0.75, 0.95]),
                },
                "top_values": values,
                "suppression": notes,
            }
        return out


# --------------------------------------------------------------------------
# chunked readers -- a whole table is never held in memory
# --------------------------------------------------------------------------

def read_header(path, entry, fmt):
    """Column names as they appear in the delivery file."""
    if entry["extension"] in (".parquet", ".pqt"):
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema_arrow.names)
    delimiter = fmt.get("delimiter") or "|"
    with open_text(path, entry["compression"]) as handle:
        line = handle.readline().rstrip("\r\n")
    if fmt.get("has_header") is False:
        return None
    return [c.strip().strip('"') for c in line.split(delimiter)]


def iter_chunks(path, entry, fmt, names, chunk_rows):
    """Yield string-typed DataFrames from one delivery file."""
    if entry["extension"] in (".parquet", ".pqt"):
        yield from _iter_parquet(path, chunk_rows)
        return
    yield from _iter_delimited(path, fmt, names, chunk_rows)


def _iter_parquet(path, chunk_rows):
    if duckdb is not None:
        try:
            reader = duckdb.sql(
                "SELECT * FROM read_parquet(?)", params=[path]
            ).fetch_record_batch(chunk_rows)
            for batch in reader:
                yield batch.to_pandas()
            return
        except Exception:  # noqa: BLE001 - fall back to pyarrow
            pass
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=chunk_rows):
        yield batch.to_pandas()


def _iter_delimited(path, fmt, names, chunk_rows):
    delimiter = fmt.get("delimiter") or "|"
    header = 0 if fmt.get("has_header") is not False else None
    kwargs = dict(sep=delimiter, dtype=str, header=header, chunksize=chunk_rows,
                  na_filter=False, quotechar=fmt.get("quotechar") or '"',
                  on_bad_lines="skip", encoding="utf-8", encoding_errors="replace",
                  compression="infer")
    if header is None and names:
        kwargs["names"] = names
    try:
        for chunk in pd.read_csv(path, **kwargs):
            yield chunk
        return
    except Exception:  # noqa: BLE001 - ragged or badly quoted files
        pass
    # Python engine tolerates ragged rows that the C parser rejects.
    kwargs.update(engine="python", quoting=csv.QUOTE_NONE)
    kwargs.pop("encoding_errors", None)
    for chunk in pd.read_csv(path, **kwargs):
        yield chunk


# --------------------------------------------------------------------------
# cross-table integrity
# --------------------------------------------------------------------------

def foreign_keys(cfg):
    """child column -> name of the table whose primary key it should point at."""
    out = {}
    for field_role, parent_role in roles.PARENT_OF.items():
        parent = cfg.table(parent_role)
        for role in cfg.table_roles():
            if field_role in roles.FIELDS[role]:
                out[cfg.field(role, field_role)] = parent
    return out


def hash_keys(series):
    """Stable 64-bit hashes; the key values themselves are never retained."""
    text = series.astype("string").dropna()
    if not len(text):
        return np.empty(0, dtype="uint64")
    return pd.util.hash_array(text.to_numpy(dtype=object))


class KeyCollector:
    """Holds hashed key columns so orphan rates can be computed at the end."""

    def __init__(self):
        self.buffers = {}

    def add(self, table, column, values):
        self.buffers.setdefault((table, column), []).append(values)

    def compress(self, table):
        for key in [k for k in self.buffers if k[0] == table]:
            blocks = self.buffers[key]
            if len(blocks) > 1:
                self.buffers[key] = [np.concatenate(blocks)]

    def get(self, table, column):
        blocks = self.buffers.get((table, column))
        return np.concatenate(blocks) if blocks else np.empty(0, dtype="uint64")


def integrity_report(cfg, collector, tables_seen):
    """Counts of people, orphan-key rates and rows per person, all as counts."""
    person_key = cfg.field(roles.PERSON, "person_id")
    parents = foreign_keys(cfg)
    report = {"patients_per_table": {}, "rows_per_patient": {}, "orphan_keys": {}}
    for table in tables_seen:
        keys = collector.get(table, person_key)
        if not keys.size:
            continue
        unique, counts = np.unique(keys, return_counts=True)
        report["patients_per_table"][table] = int(unique.size)
        report["rows_per_patient"][table] = {
            "mean": round(float(counts.mean()), 3),
            "max": int(counts.max()),
            "deciles": [int(v) for v in np.quantile(counts, np.arange(0.1, 1.0, 0.1))],
        }
    for (table, column), _ in sorted(collector.buffers.items()):
        parent = parents.get(column)
        if parent is None or parent == table or parent not in tables_seen:
            continue
        parent_key = person_key if column == person_key else column
        parent_keys = collector.get(parent, parent_key)
        child = collector.get(table, column)
        if not parent_keys.size or not child.size:
            continue
        known = np.unique(parent_keys)
        missing = ~np.isin(child, known)
        distinct_child = np.unique(child)
        report["orphan_keys"][f"{table}.{column} -> {parent}"] = {
            "child_rows_with_key": int(child.size),
            "orphan_rows": int(missing.sum()),
            "orphan_row_fraction": round(float(missing.mean()), 6),
            "distinct_child_keys": int(distinct_child.size),
            "orphan_distinct_keys": int((~np.isin(distinct_child, known)).sum()),
        }
    return report


# --------------------------------------------------------------------------
# table profiling
# --------------------------------------------------------------------------

def profile_table(table_role, entries, root, cfg, args, rng, collector, errors):
    """Stream every shard of one table and accumulate column statistics."""
    table_name = cfg.table(table_role)
    schema_fields = cfg.field_specs(table_role)
    fields = {f["name"]: f for f in schema_fields}
    codes = code_columns(cfg)
    keys_of_interest = set(foreign_keys(cfg)) | {cfg.field(roles.PERSON, "person_id"),
                                                 cfg.field(roles.ENCOUNTER, "encounter_id")}
    stats = {}
    rows = 0
    sampled = False
    file_reports = []
    header_seen = None

    for entry in entries:
        path = os.path.join(root, entry["path"])
        fmt = sniff_format(path, entry["compression"]) if entry["extension"] not in (".parquet", ".pqt") \
            else {"delimiter": None, "delimiter_name": "n/a (parquet)", "has_header": True,
                  "quotechar": None, "sample_error": None}
        record = {"path": entry["path"], "size_bytes": entry["size_bytes"],
                  "compression": entry["compression"], "format": fmt, "rows_read": 0,
                  "error": None}
        try:
            names = read_header(path, entry, fmt)
            if header_seen is None:
                header_seen = names
            for chunk in iter_chunks(path, entry, fmt, names, args.chunk_rows):
                if args.sample_rows and rows >= args.sample_rows:
                    sampled = True
                    break
                if args.sample_rows and rows + len(chunk) > args.sample_rows:
                    chunk = chunk.iloc[: args.sample_rows - rows]
                    sampled = True
                for column in chunk.columns:
                    key = str(column)
                    if key not in stats:
                        stats[key] = ColumnStats(key, fields.get(key), rng,
                                                 args.max_distinct, codes)
                    stats[key].update(chunk[column])
                    if key in keys_of_interest:
                        collector.add(table_name, key, hash_keys(chunk[column]))
                rows += len(chunk)
                record["rows_read"] += len(chunk)
                if record["rows_read"] % (args.chunk_rows * 20) == 0:
                    progress(f"    {entry['path']}: {rows:,} rows")
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors.append({"table": table_name, "path": entry["path"], "error": record["error"]})
        file_reports.append(record)
        collector.compress(table_name)

    documented = [f["name"] for f in schema_fields]
    found = header_seen or list(stats)
    reconciliation = {
        "columns_in_file": found,
        "columns_in_dictionary": documented,
        "missing_from_file": [c for c in documented if c not in found],
        "extra_in_file": [c for c in found if c not in documented],
        "column_order_matches_dictionary": list(found) == documented,
    }
    return {
        "rows_profiled": rows,
        "sampled": sampled,
        "sample_cap": args.sample_rows,
        "n_shards": len(entries),
        "total_bytes": sum(e["size_bytes"] or 0 for e in entries),
        "files": file_reports,
        "reconciliation": reconciliation,
        "columns": {name: st.to_dict(args.min_cell, args.top_n, args.max_distinct)
                    for name, st in stats.items()},
    }


# --------------------------------------------------------------------------
# disclosure guard
# --------------------------------------------------------------------------

def assert_no_disclosure(profile, min_cell):
    """Fail loudly rather than write a profile that leaks values.

    Two invariants: no id column ever carries a value list, and no listed value
    has a count below the suppression floor.
    """
    problems = []
    for table, report in profile.get("tables", {}).items():
        for column, entry in report.get("columns", {}).items():
            if entry.get("is_id_column"):
                if "categorical" in entry or "free_text" in entry:
                    problems.append(f"{table}.{column}: id column carries values")
                continue
            for section in ("categorical", "free_text"):
                block = entry.get(section) or {}
                values = block.get("values") or block.get("top_values") or []
                for item in values:
                    if item["count"] < min_cell:
                        problems.append(
                            f"{table}.{column}: value below min-cell ({item['count']})")
    if problems:
        raise SystemExit("DISCLOSURE GUARD FAILED:\n  " + "\n  ".join(problems[:50]))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="root of the delivery; no default, so it cannot run by accident")
    parser.add_argument("--out", default=os.path.join("outputs", "real_data_profile.json"))
    parser.add_argument("--config", default=schema_config.DEFAULT_PATH)
    parser.add_argument("--sample-rows", type=int, default=2_000_000,
                        help="rows to profile per table; 0 scans every row")
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument("--min-cell", type=int, default=20,
                        help="values occurring fewer times than this are suppressed")
    parser.add_argument("--top-n", type=int, default=50,
                        help="most frequent values reported for free-text columns")
    parser.add_argument("--max-distinct", type=int, default=200,
                        help="columns at or below this many distinct values are categorical")
    parser.add_argument("--include-underscore-dirs", action="store_true",
                        help="also profile directories whose name starts with '.' or '_'")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"--root is not a directory: {args.root}")
    try:
        cfg = schema_config.SchemaConfig.load(args.config)
    except schema_config.ConfigError as exc:
        raise SystemExit(str(exc)) from None
    start_banner(args)

    rng = np.random.default_rng(args.seed)
    started = time.time()

    progress("walking the delivery tree")
    entries = walk_delivery(args.root, args.include_underscore_dirs)
    groups, unmatched = match_tables(entries, cfg)
    progress(f"{len(entries)} files, {len(groups)} tables matched, {len(unmatched)} unmatched")

    errors, collector = [], KeyCollector()
    tables = {}
    for role in cfg.table_roles():
        name = cfg.table(role)
        if name not in groups:
            continue
        progress(f"profiling {name} ({len(groups[name])} file(s))")
        tables[name] = profile_table(role, groups[name], args.root, cfg,
                                     args, rng, collector, errors)
        progress(f"  {name}: {tables[name]['rows_profiled']:,} rows"
                 f"{' (sampled)' if tables[name]['sampled'] else ' (full scan)'}")

    by_table_bytes = {n: r["total_bytes"] for n, r in tables.items()}
    profile = {
        "_meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": os.path.basename(__file__),
            "root": os.path.abspath(args.root),
            "schema_source": cfg.source,
            "duckdb_available": duckdb is not None,
            "elapsed_seconds": None,
            "disclosure": {
                "content": "aggregate statistics only; no row-level records",
                "min_cell": args.min_cell,
                "top_n": args.top_n,
                "id_columns": "values never emitted for columns whose name ends in _id",
            },
            "arguments": vars(args),
        },
        "files": {
            "total_files": len(entries),
            "total_bytes": sum(e["size_bytes"] or 0 for e in entries),
            "bytes_by_table": by_table_bytes,
            "unmatched_files": unmatched,
            "skipped_files": [e["path"] for e in entries if e["skipped"]],
            "inventory": entries,
        },
        "tables": tables,
        "tables_in_dictionary_not_delivered": [cfg.table(r) for r in cfg.table_roles()
                                               if cfg.table(r) not in tables],
        "integrity": integrity_report(cfg, collector, set(tables)),
        "errors": errors,
    }
    profile["_meta"]["elapsed_seconds"] = round(time.time() - started, 1)

    assert_no_disclosure(profile, args.min_cell)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(profile, handle, indent=2, default=str)

    end_banner(args, os.path.abspath(args.out), len(tables), len(errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
