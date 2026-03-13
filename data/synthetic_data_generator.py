import numpy as np
import pandas as pd

np.random.seed(42)


def generate_synthetic_data(n_hcps=5000):
    specialties = ["Cardiology", "Neurology", "Endocrinology"]
    practice_types = ["Private", "Hospital", "Academic"]
    regions = ["Urban", "Semi-Urban", "Rural"]

    df = pd.DataFrame({
        "hcp_id": range(n_hcps),
        "specialty": np.random.choice(specialties, n_hcps),
        "practice_type": np.random.choice(practice_types, n_hcps),
        "region": np.random.choice(regions, n_hcps),
        "base_sales": np.random.normal(100, 20, n_hcps),
        "engagement_score": np.random.uniform(0, 1, n_hcps),
        "innovation_score": np.random.uniform(0, 1, n_hcps),
        "calls": np.random.poisson(3, n_hcps)
    })

    # Treatment definition
    df["treatment"] = np.where(df["calls"] >= 3, 1, 0)

    # True hidden uplift signal
    true_uplift = (
        5 * df["innovation_score"]
        + 3 * df["engagement_score"]
        - 2 * np.where(df["region"] == "Urban", 1, 0)
    )

    # Sales outcome
    df["sales"] = (
        df["base_sales"]
        + df["treatment"] * true_uplift
        + np.random.normal(0, 5, n_hcps)
    )

    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/raw/synthetic_hcp_data.csv", index=False)