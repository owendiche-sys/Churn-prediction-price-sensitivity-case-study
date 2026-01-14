
import pandas as pd
import numpy as np

CLIENT_PATH = "client_data.csv"
PRICE_PATH  = "price_data.csv"
OUTPUT_CSV  = "data_feature_engineered.csv"

MISSING_MARKERS = ["MISSING", "missing", "NA", "N/A", "-", "null", "None", ""]

def standardize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].replace(MISSING_MARKERS, np.nan)
    return df

def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_cols = [c for c in df.columns if ("date" in c.lower()) or ("time" in c.lower())]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def drop_low_value_columns(df: pd.DataFrame, missing_thresh: float = 0.90):
    df = df.copy()
    protected = set([c for c in ["id", "churn"] if c in df.columns])

    nunique = df.nunique(dropna=True)
    constant_cols = [c for c in df.columns if nunique[c] <= 1 and c not in protected]

    missing_pct = (df.isna().sum() / len(df))
    mostly_missing_cols = [c for c in df.columns if missing_pct[c] >= missing_thresh and c not in protected]

    drop_cols = sorted(list(set(constant_cols + mostly_missing_cols)))
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df

def add_date_features(df: pd.DataFrame, date_col: str, prefix: str):
    if date_col not in df.columns:
        return df
    df[prefix + "_year"] = df[date_col].dt.year
    df[prefix + "_month"] = df[date_col].dt.month
    return df

def main():
    df_client = pd.read_csv(CLIENT_PATH)
    df_price  = pd.read_csv(PRICE_PATH)

    df_client = standardize_missing_markers(df_client)
    df_price  = standardize_missing_markers(df_price)

    df_client = parse_date_columns(df_client)
    df_price  = parse_date_columns(df_price)

    df_client = drop_low_value_columns(df_client, missing_thresh=0.90)
    df_price  = drop_low_value_columns(df_price, missing_thresh=0.90)

    # reference date
    ref_candidates = []
    for col in ["date_activ", "date_end", "date_modif_prod", "date_renewal"]:
        if col in df_client.columns:
            ref_candidates.append(df_client[col].max())
    if "price_date" in df_price.columns:
        ref_candidates.append(df_price["price_date"].max())
    REF_DATE = max([d for d in ref_candidates if pd.notna(d)])

    # client FE
    df_client_fe = df_client.copy()
    for dc, px in [("date_activ","activ"), ("date_end","end"), ("date_modif_prod","modif"), ("date_renewal","renewal")]:
        df_client_fe = add_date_features(df_client_fe, dc, px)

    if "date_activ" in df_client_fe.columns:
        df_client_fe["tenure_days"] = (REF_DATE - df_client_fe["date_activ"]).dt.days
    if "date_modif_prod" in df_client_fe.columns:
        df_client_fe["days_since_modif"] = (REF_DATE - df_client_fe["date_modif_prod"]).dt.days
    if "date_renewal" in df_client_fe.columns:
        df_client_fe["days_to_renewal"] = (df_client_fe["date_renewal"] - REF_DATE).dt.days
        df_client_fe["renewal_within_30d"] = ((df_client_fe["days_to_renewal"] >= 0) & (df_client_fe["days_to_renewal"] <= 30)).astype(int)
    if "date_end" in df_client_fe.columns:
        df_client_fe["days_to_end"] = (df_client_fe["date_end"] - REF_DATE).dt.days
        df_client_fe["contract_ended"] = (df_client_fe["date_end"] <= REF_DATE).astype(int)

    if "cons_12m" in df_client_fe.columns and "cons_gas_12m" in df_client_fe.columns:
        df_client_fe["total_cons_12m"] = df_client_fe["cons_12m"].fillna(0) + df_client_fe["cons_gas_12m"].fillna(0)
    else:
        df_client_fe["total_cons_12m"] = df_client_fe.get("cons_12m", 0)

    if "cons_12m" in df_client_fe.columns:
        df_client_fe["avg_elec_monthly_cons"] = df_client_fe["cons_12m"] / 12.0

    if "total_cons_12m" in df_client_fe.columns and "cons_gas_12m" in df_client_fe.columns:
        denom = df_client_fe["total_cons_12m"].replace(0, np.nan)
        df_client_fe["gas_share_total"] = df_client_fe["cons_gas_12m"] / denom

    if "cons_last_month" in df_client_fe.columns and "avg_elec_monthly_cons" in df_client_fe.columns:
        denom = df_client_fe["avg_elec_monthly_cons"].replace(0, np.nan)
        df_client_fe["last_month_vs_avg"] = df_client_fe["cons_last_month"] / denom

    if "forecast_cons_12m" in df_client_fe.columns and "cons_12m" in df_client_fe.columns:
        df_client_fe["forecast_error_cons_12m"] = df_client_fe["forecast_cons_12m"] - df_client_fe["cons_12m"]
        denom = df_client_fe["cons_12m"].replace(0, np.nan)
        df_client_fe["forecast_ratio_cons_12m"] = df_client_fe["forecast_cons_12m"] / denom

    if "forecast_price_energy_peak" in df_client_fe.columns and "forecast_price_energy_off_peak" in df_client_fe.columns:
        df_client_fe["forecast_energy_price_spread"] = df_client_fe["forecast_price_energy_peak"] - df_client_fe["forecast_price_energy_off_peak"]

    if "margin_gross_pow_ele" in df_client_fe.columns and "margin_net_pow_ele" in df_client_fe.columns:
        df_client_fe["gross_minus_net_margin_ele"] = df_client_fe["margin_gross_pow_ele"] - df_client_fe["margin_net_pow_ele"]

    if "net_margin" in df_client_fe.columns and "nb_prod_act" in df_client_fe.columns:
        denom = df_client_fe["nb_prod_act"].replace(0, np.nan)
        df_client_fe["net_margin_per_product"] = df_client_fe["net_margin"] / denom

    if "pow_max" in df_client_fe.columns and "cons_12m" in df_client_fe.columns:
        denom = df_client_fe["cons_12m"].replace(0, np.nan)
        df_client_fe["powmax_per_elec_cons"] = df_client_fe["pow_max"] / denom

    # price FE
    df_price_fe = df_price.copy()
    df_price_fe = df_price_fe.sort_values(["id", "price_date"])
    price_cols = [c for c in df_price_fe.columns if c not in ["id","price_date"]]

    agg_dict = {c: ["mean","std","min","max"] for c in price_cols}
    price_agg = df_price_fe.groupby("id").agg(agg_dict)
    price_agg.columns = ["_".join(col) for col in price_agg.columns.values]
    price_agg = price_agg.reset_index()

    last_rows = df_price_fe.groupby("id").tail(1).set_index("id")
    first_rows = df_price_fe.groupby("id").head(1).set_index("id")

    latest_feats = last_rows[price_cols].add_prefix("latest_").reset_index()
    earliest_feats = first_rows[price_cols].add_prefix("earliest_").reset_index()

    change_df = pd.DataFrame({"id": last_rows.index})
    for c in price_cols:
        latest = last_rows[c]
        earliest = first_rows[c]
        change_df[f"chg_{c}"] = (latest - earliest).values
        change_df[f"pct_chg_{c}"] = ((latest - earliest) / earliest.replace(0, np.nan)).values

    # spreads on latest
    latest_spreads = latest_feats.copy()
    if "latest_price_peak_var" in latest_spreads.columns and "latest_price_off_peak_var" in latest_spreads.columns:
        latest_spreads["latest_spread_peak_offpeak_var"] = latest_spreads["latest_price_peak_var"] - latest_spreads["latest_price_off_peak_var"]
    if "latest_price_mid_peak_var" in latest_spreads.columns and "latest_price_off_peak_var" in latest_spreads.columns:
        latest_spreads["latest_spread_mid_offpeak_var"] = latest_spreads["latest_price_mid_peak_var"] - latest_spreads["latest_price_off_peak_var"]
    if "latest_price_peak_fix" in latest_spreads.columns and "latest_price_off_peak_fix" in latest_spreads.columns:
        latest_spreads["latest_spread_peak_offpeak_fix"] = latest_spreads["latest_price_peak_fix"] - latest_spreads["latest_price_off_peak_fix"]
    if "latest_price_mid_peak_fix" in latest_spreads.columns and "latest_price_off_peak_fix" in latest_spreads.columns:
        latest_spreads["latest_spread_mid_offpeak_fix"] = latest_spreads["latest_price_mid_peak_fix"] - latest_spreads["latest_price_off_peak_fix"]

    six_months_cut = REF_DATE - pd.Timedelta(days=183)
    df_last6 = df_price_fe[df_price_fe["price_date"] >= six_months_cut].copy()
    agg6_dict = {c: ["mean","std"] for c in price_cols}
    price_6m = df_last6.groupby("id").agg(agg6_dict)
    price_6m.columns = [f"6m_{a}_{b}" for (a, b) in price_6m.columns]
    price_6m = price_6m.reset_index()

    price_features = price_agg.merge(latest_feats, on="id", how="left") \
                              .merge(earliest_feats, on="id", how="left") \
                              .merge(change_df, on="id", how="left") \
                              .merge(latest_spreads[["id"] + [c for c in latest_spreads.columns if c.startswith("latest_spread_")]], on="id", how="left") \
                              .merge(price_6m, on="id", how="left")

    df_merged = df_client_fe.merge(price_features, on="id", how="left")

    # drop datetime cols
    datetime_cols = df_merged.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    df_final = df_merged.drop(columns=datetime_cols, errors="ignore")

    # fill numeric NaNs
    num_cols = df_final.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if c != "churn":
            df_final[c] = df_final[c].fillna(df_final[c].median())

    df_final.to_csv(OUTPUT_CSV, index=False)
    print("Saved:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
