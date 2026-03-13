import pandas as pd
import numpy as np


def qini_curve(df, uplift_col="uplift_score", treatment_col="treatment", outcome_col="sales"):
    """
    Computes data points for the Qini curve.
    """
    df_sorted = df.sort_values(by=uplift_col, ascending=False).reset_index(drop=True)
    df_sorted["n_treated"] = df_sorted[treatment_col].cumsum()
    df_sorted["n_control"] = (~df_sorted[treatment_col].astype(bool)).cumsum()

    # Cumulative sales (outcomes)
    df_sorted["cum_y_treated"] = (df_sorted[outcome_col] * df_sorted[treatment_col]).cumsum()
    df_sorted["cum_y_control"] = (df_sorted[outcome_col] * (~df_sorted[treatment_col].astype(bool))).cumsum()

    # Qini computation
    # Q(t) = Y_t - Y_c * (N_t / N_c)
    # Handle division by zero
    n_c_safe = df_sorted["n_control"].replace(0, 1)
    df_sorted["qini"] = df_sorted["cum_y_treated"] - df_sorted["cum_y_control"] * (df_sorted["n_treated"] / n_c_safe)

    # Prepend 0 to start at origin
    qini_values = np.append(0, df_sorted["qini"].values)
    population = np.append(0, len(df_sorted.index)) # Need percentages instead: np.arange(len(df_sorted) + 1) / len(df_sorted)
    population_pct = np.arange(len(df_sorted) + 1) / len(df_sorted)
    
    return population_pct, qini_values

def calculate_auuc(population_pct, qini_values):
    """
    Calculates the Area Under the Uplift Curve (AUUC).
    """
    return np.trapezoid(qini_values, population_pct)
