import pandas as pd

csv_path = "day3_estimated_7d_ver2.csv"
df = pd.read_csv(csv_path)

#print(df[["category", "is_short", "CPI"]].head())
#print(df[["category", "is_short", "CPI"]].isna().sum())

def mark_iqr_outlier(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

category_outlier_pct = (
    df
    .groupby("category")
    .apply(lambda g: pd.Series({
        "total_rows": len(g),
        "outlier_rows": mark_iqr_outlier(g["CPI"]).sum(),
        "outlier_pct": mark_iqr_outlier(g["CPI"]).mean() * 100
    }))
    .reset_index()
    .sort_values("outlier_pct", ascending=False)
)

print(category_outlier_pct)

is_short_outlier_pct = (
    df
    .groupby("is_short")
    .apply(lambda g: pd.Series({
        "total_rows": len(g),
        "outlier_rows": mark_iqr_outlier(g["CPI"]).sum(),
        "outlier_pct": mark_iqr_outlier(g["CPI"]).mean() * 100
    }))
    .reset_index()
)

print(is_short_outlier_pct)
