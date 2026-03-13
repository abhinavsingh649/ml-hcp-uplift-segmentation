import pandas as pd


def encode_categoricals(df):
    df = pd.get_dummies(
        df,
        columns=["specialty", "practice_type", "region"],
        drop_first=True
    )
    return df


def create_features(df):
    df["call_intensity"] = df["calls"] / (df["base_sales"] + 1)
    df["engagement_x_innovation"] = (
        df["engagement_score"] * df["innovation_score"]
    )
    return df


def prepare_model_data(df):
    df = encode_categoricals(df)
    df = create_features(df)
    return df