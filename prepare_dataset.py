# prepare_dataset.py
# ------------------------------------------------------------
# Step 1 – Data Preparation & Feature Engineering
# Generates a clean dataset with the following columns:
# timestamp(datetime), source_ip, destination_ip, port, protocol,
# packet_size, attack_type, source_ip_valid, destination_ip_valid,
# hour, day, weekday, time_segment(morning/afternoon/night), is_weekend
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import ipaddress

# ========== CONFIG ==========
INPUT_CSV  = "Smart Grid Intrusion Detection Dataset - Copy.csv"
OUTPUT_CSV = "processed_smart_grid_attacks.csv"
TIMEZONE   = None  # e.g., "Europe/Madrid" if you want local time conversion

print("Loading CSV...")

# ===== Robust CSV reading =====
try:
    df = pd.read_csv(
        INPUT_CSV,
        sep=",",
        encoding="utf-8-sig",
        quotechar='"',
        low_memory=False,
    )
except ValueError as e:
    print("Retrying with engine='python' due to:", e)
    df = pd.read_csv(
        INPUT_CSV,
        sep=",",
        encoding="utf-8-sig",
        quotechar='"',
        engine="python",
        on_bad_lines="skip",
    )

df.columns = [c.strip() for c in df.columns]

# ===== Column candidates detection =====
def pick(candidates: list[str], df_cols: list[str]):
    """Return the first matching column name from the dataset."""
    lc = {c.lower(): c for c in df_cols}
    for name in candidates:
        if name.lower() in lc:
            return lc[name.lower()]
    return None

timestamp_col = pick(["timestamp", "time", "datetime", "date_time", "event_time"], df.columns)
if not timestamp_col:
    raise ValueError("Timestamp column not found. Please rename it or adjust candidate names.")

src_ip_col = pick(["source_ip", "src_ip", "ip_src", "srcip"], df.columns)
dst_ip_col = pick(["destination_ip", "dst_ip", "ip_dst", "dstip"], df.columns)
port_col   = pick(["destination_port", "dst_port", "port", "src_port"], df.columns)
proto_col  = pick(["protocol", "proto", "transport"], df.columns)
size_col   = pick(["packet_size", "size", "length", "len", "bytes"], df.columns)
attack_col = pick(["attack_type", "attack", "label", "class"], df.columns)

# ===== Basic cleaning =====
before = len(df)
df = df.drop_duplicates()
df = df[~df[timestamp_col].isna()].copy()
print(f"Initial rows: {before} | After cleaning: {len(df)}")

# ===== Timestamp conversion =====
df["timestamp_dt"] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
if df["timestamp_dt"].isna().mean() > 0.10:
    tmp = pd.to_datetime(df[timestamp_col], errors="coerce")
    if tmp.notna().sum() > df["timestamp_dt"].notna().sum():
        df["timestamp_dt"] = tmp

df = df[df["timestamp_dt"].notna()].copy()

# Optional: convert timezone
if TIMEZONE is not None and pd.api.types.is_datetime64tz_dtype(df["timestamp_dt"].dtype):
    df["timestamp_dt"] = df["timestamp_dt"].dt.tz_convert(TIMEZONE)

# ===== Temporal features =====
ts = df["timestamp_dt"]
df["hour"] = ts.dt.hour
df["day"] = ts.dt.day
df["weekday"] = ts.dt.weekday  # Monday=0, Sunday=6

def time_segment_from_hour(h: int) -> str:
    if 6 <= h <= 11:
        return "morning"
    elif 12 <= h <= 19:
        return "afternoon"
    return "night"

df["time_segment"] = df["hour"].apply(time_segment_from_hour)
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

# ===== Normalize attack labels =====
if attack_col:
    df["attack_type_raw"] = df[attack_col].astype(str).str.strip().str.lower()
    canonical_map = {
        "dos": "dos", "denial of service": "dos", "ddos": "dos",
        "normal": "benign", "benign": "benign",
        "mitm": "mitm", "man-in-the-middle": "mitm",
        "malware": "malware", "apt": "apt",
        "sql injection": "sql_injection", "sql_injection": "sql_injection",
        "probe": "probe", "scan": "scan",
        "ransomware": "ransomware", "phishing": "phishing", "injection": "injection",
    }
    df["attack_type"] = df["attack_type_raw"].map(lambda x: canonical_map.get(x, x))
else:
    df["attack_type"] = np.nan

# ===== Standardize and rename columns =====
df["timestamp"] = df["timestamp_dt"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")

df["source_ip"] = df[src_ip_col] if src_ip_col else np.nan
df["destination_ip"] = df[dst_ip_col] if dst_ip_col else np.nan
df["port"] = pd.to_numeric(df[port_col], errors="coerce") if port_col else np.nan
df["protocol"] = df[proto_col].astype(str).str.strip() if proto_col else np.nan
df["packet_size"] = pd.to_numeric(df[size_col], errors="coerce") if size_col else np.nan

# ===== IP validation =====
def is_valid_ip(x) -> bool:
    try:
        ipaddress.ip_address(str(x))
        return True
    except Exception:
        return False

df["source_ip_valid"] = df["source_ip"].apply(is_valid_ip).astype(int)
df["destination_ip_valid"] = df["destination_ip"].apply(is_valid_ip).astype(int)

# ===== Final column selection =====
final_cols = [
    "timestamp",
    "source_ip",
    "destination_ip",
    "port",
    "protocol",
    "packet_size",
    "attack_type",
    "source_ip_valid",
    "destination_ip_valid",
    "hour",
    "day",
    "weekday",
    "time_segment",
    "is_weekend",
]

# Ensure all exist (fill missing with NaN)
for c in final_cols:
    if c not in df.columns:
        df[c] = np.nan

df_out = df[final_cols].copy()

# ===== Save result =====
out_path = Path(OUTPUT_CSV)
df_out.to_csv(out_path, index=False)

print("\n✅ Done.")
print("Output saved to:", out_path.resolve())
print("Rows:", len(df_out), "| Columns:", len(df_out.columns))
print("Final columns:", final_cols)
