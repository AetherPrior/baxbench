# ===== Metrics computation: correlations vs pass/security =====
from typing import Tuple, Dict, Any, List, Optional
import json
import os
import pandas as pd
import numpy as np
from concept_eval_batched import ID_COL, PASS_COL, SEC_COL, ACTION_DEFS_SEC

try:
    from scipy.stats import pearsonr as _pearsonr
except Exception:
    _pearsonr = None  # p-values will be None if SciPy isn't installed

def _latest_per_id_action(cache_jsonl: str) -> pd.DataFrame:
    """
    Read the append-only cache JSONL and keep the *last* record per (Id, action_key).
    Returns columns: Id, action_key, presence_str, quality
    """
    if not cache_jsonl or not os.path.exists(cache_jsonl):
        # empty frame with correct dtypes
        return pd.DataFrame(columns=["Id", "action_key", "presence_str", "quality"])

    records: List[Dict[str, Any]] = []
    with open(cache_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # expected keys in your writer: Id, action_key, presence, quality, ...
            rid = str(rec.get("Id"))
            ak  = rec.get("action_key")
            pres = rec.get("presence")   # "Yes"/"No"
            qual = rec.get("quality")    # "Poor"/"Adequate"/"Strong" or None
            if rid is None or ak is None:
                continue
            records.append({"Id": rid, "action_key": ak, "presence_str": pres, "quality": qual})

    if not records:
        return pd.DataFrame(columns=["Id", "action_key", "presence_str", "quality"])

    df = pd.DataFrame(records)
    # keep last occurrence per (Id, action_key)
    df = df.groupby(["Id", "action_key"], as_index=False).tail(1).reset_index(drop=True)
    return df

def _wide_action_flags_from_cache(cache_df: pd.DataFrame, action_keys: List[str]) -> pd.DataFrame:
    """
    Pivot cache rows into wide columns for each action:
    - {KEY}_presence ∈ {0,1}
    - {KEY}_quality   str or NaN
    - {KEY}_strong_gated ∈ {0,1}  (1 iff presence==1 and quality=="Strong")
    Returns: columns [Id, ...derived columns...]
    """
    if cache_df.empty:
        out = pd.DataFrame({"Id": []})
        for k in action_keys:
            out[f"{k}_presence"] = []
            out[f"{k}_quality"] = []
            out[f"{k}_strong_gated"] = []
        return out

    cache_df = cache_df.copy()
    cache_df["presence"] = np.where(cache_df["presence_str"].str.lower() == "yes", 1, 0)

    # Build one row per Id with all keys
    rows: Dict[str, Dict[str, Any]] = {}
    for rid, ak, pres, qual in cache_df[["Id", "action_key", "presence", "quality"]].itertuples(index=False):
        rid = str(rid)
        rows.setdefault(rid, {"Id": rid})
        rows[rid][f"{ak}_presence"] = int(pres)
        rows[rid][f"{ak}_quality"] = qual if pd.notna(qual) else None
        rows[rid][f"{ak}_strong_gated"] = int(pres == 1 and str(qual) == "Strong")

    # Ensure all columns exist for all actions
    for r in rows.values():
        for ak in action_keys:
            r.setdefault(f"{ak}_presence", 0)
            r.setdefault(f"{ak}_quality", None)
            r.setdefault(f"{ak}_strong_gated", 0)

    wide = pd.DataFrame(list(rows.values()))
    # Consistent column order (optional)
    cols = ["Id"]
    for ak in action_keys:
        cols += [f"{ak}_presence", f"{ak}_quality", f"{ak}_strong_gated"]
    wide = wide.reindex(columns=cols)
    return wide

def _pearson(x: pd.Series, y: pd.Series) -> Tuple[float, Optional[float], int]:
    """
    Compute Pearson r (and p if SciPy available) on pairwise non-null rows.
    Returns: (r, p_value_or_None, n)
    """
    s = pd.concat([x, y], axis=1).dropna()
    n = len(s)
    if n < 2:
        return (float("nan"), None, n)
    x_, y_ = s.iloc[:, 0].astype(float), s.iloc[:, 1].astype(float)

    if _pearsonr is not None:
        r, p = _pearsonr(x_, y_)
        return (float(r), float(p), n)
    else:
        # fallback: r only
        r = float(np.corrcoef(x_, y_)[0, 1])
        return (r, None, n)

def compute_action_metrics(
    csv_path: str,
    cache_jsonl: str,
    id_col: str = ID_COL,
    pass_col: str = PASS_COL,
    sec_col: str = SEC_COL,
    action_defs: Dict[str, Tuple[str, str]] = ACTION_DEFS_SEC,
    save_cor_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads CSV + cache JSONL, derives action presence/quality/strong_gated,
    and computes Pearson correlations vs pass & sec.

    Returns:
      - merged_df: the full merged DataFrame (one row per example)
      - corr_presence_df: tidy table of correlations for presence vs {pass,sec}
      - corr_strong_gated_df: tidy table of correlations for strong_gated vs {pass,sec}
    """
    # Load main CSV
    try: 
        base = pd.read_csv(csv_path) # .iloc[:200]
    except:
        base = pd.read_csv(csv_path, sep='\t')
    
    # 200 rows, shuffled with random seed = 42
    # base = base.iloc[:100] # --- IGNORE ---
    base = base.sample(frac=1.0, random_state=42).reset_index(drop=True) # --- IGNORE ---
    if id_col not in base.columns:
        raise KeyError(f"Missing id column {id_col!r} in CSV")
    if pass_col not in base.columns or sec_col not in base.columns:
        raise KeyError(f"CSV must include {pass_col!r} and {sec_col!r} float columns")

    # Read latest per (Id, action_key) from cache
    cache_df = _latest_per_id_action(cache_jsonl)

    # Build wide flags (presence/quality/strong_gated)
    action_keys = list(action_defs.keys())
    wide_flags = _wide_action_flags_from_cache(cache_df, action_keys)

    # Left-join onto base by id
    base = base.copy()
    base[id_col] = base[id_col].astype(str)
    wide_flags["Id"] = wide_flags["Id"].astype(str)
    merged = base.merge(wide_flags, how="left", left_on=id_col, right_on="Id")

    # If some examples never appeared in cache, fill NaNs with 0/None consistently
    for ak in action_keys:
        merged[f"{ak}_presence"] = merged[f"{ak}_presence"].fillna(0).astype(int)
        merged[f"{ak}_strong_gated"] = merged[f"{ak}_strong_gated"].fillna(0).astype(int)
        # quality stays object; keep NaN if unknown
        if f"{ak}_quality" in merged.columns:
            merged[f"{ak}_quality"] = merged[f"{ak}_quality"].where(merged[f"{ak}_quality"].notna(), None)

    # Compute correlations
    rows_presence: List[Dict[str, Any]] = []
    rows_strong: List[Dict[str, Any]] = []

    for ak in action_keys:
        # presence correlations
        r_pass, p_pass, n_pass = _pearson(merged[f"{ak}_presence"], merged[pass_col])
        r_sec,  p_sec,  n_sec  = _pearson(merged[f"{ak}_presence"], merged[sec_col])

        # compute frequency of action presence
        presence_count = merged[f"{ak}_presence"].sum()

        rows_presence += [
            {"action_key": ak, "target": "pass", "metric": "presence", "r": r_pass, "p": p_pass, "n": n_pass, "presence_count": presence_count},
            {"action_key": ak, "target": "sec",  "metric": "presence", "r": r_sec,  "p": p_sec,  "n": n_sec, "presence_count": presence_count},
        ]
        # strong_gated correlations (B: gated binary)
        r_pass_s, p_pass_s, n_pass_s = _pearson(merged[f"{ak}_strong_gated"], merged[pass_col])
        r_sec_s,  p_sec_s,  n_sec_s  = _pearson(merged[f"{ak}_strong_gated"], merged[sec_col])

        # compute frequency of strong_gated
        strong_gated_count = merged[f"{ak}_strong_gated"].sum()

        rows_strong += [
            {"action_key": ak, "target": "pass", "metric": "strong_gated", "r": r_pass_s, "p": p_pass_s, "n": n_pass_s, "strong_gated_count": strong_gated_count},
            {"action_key": ak, "target": "sec",  "metric": "strong_gated", "r": r_sec_s,  "p": p_sec_s,  "n": n_sec_s, "strong_gated_count": strong_gated_count},
        ]



    corr_presence_df = pd.DataFrame(rows_presence).sort_values(["metric", "target", "action_key"]).reset_index(drop=True)
    corr_strong_df   = pd.DataFrame(rows_strong).sort_values(["metric", "target", "action_key"]).reset_index(drop=True)

    if save_cor_csv:
        # write two CSVs with clear suffixes
        root, ext = os.path.splitext(save_cor_csv)
        breakpoint()
        corr_presence_df.to_csv(f"{root}_presence{ext or '.csv'}", index=False, float_format='%.2f')
        corr_strong_df.to_csv(f"{root}_strong_gated{ext or '.csv'}", index=False, float_format='%.2f')

    return merged, corr_presence_df, corr_strong_df

# ===== Example (comment out in library use) =====
merged, corr_presence, corr_strong = compute_action_metrics(
    csv_path="./intervention_test_none/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-None_all_analysis.csv",
    cache_jsonl="cache_judge/v4_gpt-4o_full_def_judge_deepseek.jsonl",
    id_col=ID_COL,
    pass_col=PASS_COL,
    sec_col=SEC_COL,
    action_defs=ACTION_DEFS_SEC,
    save_cor_csv="results/v4_gpt-4o_action_correlations.csv",
)
print(corr_presence)
print(corr_strong)
