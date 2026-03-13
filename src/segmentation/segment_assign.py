import pandas as pd


def assign_segments(df):
    df["percentile"] = df["uplift_score"].rank(pct=True)

    df["segment"] = pd.cut(
        df["percentile"],
        bins=[0, 0.33, 0.67, 1],
        labels=["Low", "Medium", "High"]
    )

    return df