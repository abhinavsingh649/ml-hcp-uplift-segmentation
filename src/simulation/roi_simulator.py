def simulate_roi(df, top_fraction=0.3):
    df = df.sort_values(by="uplift_score", ascending=False)

    top_n = int(len(df) * top_fraction)

    top_group = df.head(top_n)
    bottom_group = df.tail(top_n)

    incremental_gain = top_group["uplift_score"].sum()
    lost_gain = bottom_group["uplift_score"].sum()

    return {
        "gain_from_top": incremental_gain,
        "gain_lost_bottom": lost_gain,
        "net_impact": incremental_gain - lost_gain
    }   