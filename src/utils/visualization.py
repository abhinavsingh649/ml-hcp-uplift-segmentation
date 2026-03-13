import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_qini_curve(population_pct, qini_values, output_path="reports/qini_curve.png"):
    """
    Plots and saves the Qini curve.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Qini curve
    plt.plot(population_pct, qini_values, label="Uplift Model Qini", color='blue', linewidth=2)
    
    # Plot random baseline
    random_base = np.linspace(0, qini_values[-1], len(population_pct))
    plt.plot(population_pct, random_base, label="Random Assignment", color='red', linestyle='--')
    
    plt.title("Qini Curve (Cumulative Uplift)")
    plt.xlabel("Fraction of Population Targeted")
    plt.ylabel("Cumulative Incremental Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Qini curve saved to {output_path}")
    plt.close()


def plot_segment_distribution(df, segment_col="segment", output_path="reports/segment_distribution.png"):
    """
    Plots and saves the distribution of predicted segments.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x=segment_col, order=["High", "Medium", "Low"], palette="viridis")
    
    plt.title("HCP Segment Distribution")
    plt.xlabel("Uplift Segment")
    plt.ylabel("Number of HCPs")
    
    # Add counts above bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
                    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Segment distribution saved to {output_path}")
    plt.close()
