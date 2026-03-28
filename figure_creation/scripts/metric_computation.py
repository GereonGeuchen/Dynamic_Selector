import pandas as pd

def print_performance_per_fid(df: pd.DataFrame):
    for fid in df["fid"].unique():
        df_fid = df[df["fid"] == fid]
        fid_sbs_sum = df_fid["sbs_precision"].sum()
        fid_vbs_sum = df_fid["vbs_precisions"].sum()
        df_kostovska_sum = df_fid["static_B150"].sum() # Kostovska's algorithm is static with budget 150
        df_selector_sum = df_fid["selector_precision"].sum()

        # Avoid division by zero when calculating closed gap
        if fid_sbs_sum - fid_sbs_sum == 0:
            if df_selector_sum - df_kostovska_sum == 0:
                closed_gap = 1.0
            else:                
                closed_gap = 0.0
        else:
            closed_gap = (df_selector_sum - fid_sbs_sum) / (fid_vbs_sum - fid_sbs_sum)

        # Avoid division by zero when calculating speedup against Kostovska
        if df_kostovska_sum == 0:
            if df_selector_sum == 0:
                speedup_kostovska = 1.0
            else:
                speedup_kostovska = float('inf')  # Infinite speedup if selector has non-zero precision while Kostovska has zero
        else:
            speedup_kostovska = df_selector_sum / df_kostovska_sum

        # Avoid division by zero when calculating speedup against SBS
        if fid_sbs_sum == 0:
            if df_selector_sum == 0:
                speedup_sbs = 1.0
            else:
                speedup_sbs = float('inf')  # Infinite speedup if selector has non-zero precision while SBS has zero
        else:
            speedup_sbs = df_selector_sum / fid_sbs_sum

        # Avoid division by zero when calculating min-max gap closed against Kostovska
        if fid_vbs_sum - df_kostovska_sum == 0:
            if df_selector_sum - df_kostovska_sum == 0:
                min_max_gap_closed_kostovska = 1.0
            else:
                min_max_gap_closed_kostovska = 0.0
        else:
            min_max_gap_closed_kostovska = (df_selector_sum - df_kostovska_sum) / (fid_vbs_sum - df_kostovska_sum)

        print(f"=== Performance for fid {fid} ===")
        print(f"Selector total precision: {df_fid['selector_precision'].sum():.4f}")
        print(f"Selector closed gap: {closed_gap:.4f}")
        print(f"Selector speedup against Kostovska: {speedup_kostovska:.4f}")
        print(f"Selector speedup against SBS: {speedup_sbs:.4f}")
        print(f"Selector min-max gap closed against Kostovska: {min_max_gap_closed_kostovska:.4f}")
        print(f"Selector mean precision: {df_fid['selector_precision'].mean():.4f}")
        print(f"Kostovska mean precision: {df_fid['static_B150'].mean():.4f}")
        print(f"SBS mean precision: {df_fid['sbs_precision'].mean():.4f}")

if __name__== "__main__":
    df = pd.read_csv(f"../data/selector_performance_data/selector_results_with_lookahead_all_epms_10_sbs.csv")
    print_performance_per_fid(df)