# prepare_dataset.py
import pandas as pd

# path to the file you uploaded
input_path = "data/pjm_hourly.csv"
output_path = "data/pjm_clean.csv"

df = pd.read_csv(input_path, parse_dates=["Datetime"], infer_datetime_format=True)
# standardize column names (some files use 'MW' or 'Load')
col_candidates = [c for c in df.columns if c.lower() in ["mw","load","value","mw/h","load_mw","pjm_load"]]
if not col_candidates:
    # try common name 'MW'
    if "MW" in df.columns:
        col = "MW"
    elif "MW/h" in df.columns:
        col = "MW/h"
    else:
        # fallback to second column
        col = df.columns[1]
else:
    col = col_candidates[0]

df = df.rename(columns={col: "MW"})
# keep only Datetime and MW (and optional Zone)
keep_cols = ["Datetime", "MW"]
if "Zone" in df.columns:
    keep_cols.append("Zone")
df = df[keep_cols]
df = df.sort_values("Datetime").reset_index(drop=True)

# derive Total_MW (if multiple zones not present it's the same)
if "Zone" in df.columns:
    total = df.groupby("Datetime")["MW"].sum().reset_index().rename(columns={"MW":"Total_MW"})
else:
    total = df[["Datetime","MW"]].rename(columns={"MW":"Total_MW"})

df = df.merge(total, on="Datetime", how="left")

# Save cleaned file
df.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to {output_path}")
print(df.head())
