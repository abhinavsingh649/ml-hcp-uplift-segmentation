import os
import pandas as pd
from src.data.feature_engineering import prepare_model_data
from src.modeling.uplift_xlearner import XLearnerUplift
from src.segmentation.segment_assign import assign_segments
from src.simulation.roi_simulator import simulate_roi
from src.utils.metrics import qini_curve, calculate_auuc
from src.utils.visualization import plot_qini_curve, plot_segment_distribution
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, REPORTS_DIR, MODELS_DIR, MODEL_PATH

def train_pipeline(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    y = df["sales"]
    treatment = df["treatment"]

    print("Engineering features...")
    df = prepare_model_data(df)

    X = df.drop(columns=["hcp_id", "sales", "treatment"])

    print("Training X-Learner Uplift Model...")
    model = XLearnerUplift()
    model.fit(X, treatment, y)

    print("Predicting uplift scores...")
    df["uplift_score"] = model.predict_uplift(X)

    return df, model

if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH.parent, exist_ok=True)

    df, model = train_pipeline(RAW_DATA_PATH)
    
    print("\nAssigning segments...")
    df = assign_segments(df)
    
    print("\nSimulating ROI...")
    roi = simulate_roi(df)
    for k, v in roi.items():
        print(f"  {k}: {v:.2f}")

    print("\nCalculating Qini curve and AUUC...")
    population_pct, qini_values = qini_curve(df)
    auuc = calculate_auuc(population_pct, qini_values)
    print(f"  AUUC: {auuc:.4f}")

    print("\nGenerating visualizations...")
    plot_qini_curve(population_pct, qini_values, output_path=os.path.join(REPORTS_DIR, "qini_curve.png"))
    plot_segment_distribution(df, output_path=os.path.join(REPORTS_DIR, "segment_distribution.png"))

    print(f"\nSaving processed data to {PROCESSED_DATA_PATH}...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)

    print("Pipeline completed successfully!")