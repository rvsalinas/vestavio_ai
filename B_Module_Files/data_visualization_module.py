"""
data_visualization_module.py

Absolute File Path (Example):
/Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/data_visualization_module.py

PURPOSE:
  - Create and save common data visualizations (time series plots, histograms, correlation heatmaps).
  - Useful for exploratory data analysis and reporting model insights.

NOTES:
  - Saves plots to a specified directory (default: "visualizations").
  - Uses matplotlib and seaborn for plotting.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Optional, List

class DataVisualization:
    """
    A class for creating data visualizations like time series, histograms, 
    and correlation heatmaps, saving them to disk for further analysis or 
    integration into reports/dashboards.
    """

    def __init__(self, save_dir: str = "visualizations"):
        """
        :param save_dir: Directory to save output plots.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        logging.info(f"[DataVisualization] Initialized with save_dir={self.save_dir}")

    def plot_time_series(self, df: pd.DataFrame, time_col: str, value_col: str, 
                         title: str = "Time Series", x_label: str = "", y_label: str = ""):
        """
        Create a basic time series plot of one numeric column over time.

        :param df: Pandas DataFrame containing the data.
        :param time_col: Column name representing time or x-axis values.
        :param value_col: Column name representing the numeric values to plot.
        :param title: Title for the plot.
        :param x_label: X-axis label (optional).
        :param y_label: Y-axis label (optional).
        """
        if time_col not in df.columns or value_col not in df.columns:
            logging.error(f"[DataVisualization] Invalid columns: {time_col} or {value_col} not found in DataFrame.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(df[time_col], df[value_col], marker='o', linestyle='-', color='blue', label=value_col)
        plt.title(title)
        plt.xlabel(x_label if x_label else time_col)
        plt.ylabel(y_label if y_label else value_col)
        plt.legend()
        
        filename = f"time_series_{value_col}.png"
        output_path = os.path.join(self.save_dir, filename)
        plt.savefig(output_path)
        plt.close()
        
        logging.info(f"[DataVisualization] Time series plot saved to {output_path}")

    def plot_histogram(self, df: pd.DataFrame, col: str, bins: int = 30, kde: bool = True):
        """
        Plot a histogram (and optional KDE) for a single numeric column.

        :param df: Pandas DataFrame containing data.
        :param col: Column name to plot the histogram for.
        :param bins: Number of bins in the histogram.
        :param kde: Whether to add a kernel density estimate line.
        """
        if col not in df.columns:
            logging.error(f"[DataVisualization] Column '{col}' not found in DataFrame.")
            return

        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, bins=bins, kde=kde, color='green')
        plt.title(f"Histogram of {col}")

        filename = f"hist_{col}.png"
        output_path = os.path.join(self.save_dir, filename)
        plt.savefig(output_path)
        plt.close()
        
        logging.info(f"[DataVisualization] Histogram for '{col}' saved to {output_path}")

    def plot_correlation(self, df: pd.DataFrame, cols: Optional[List[str]] = None):
        """
        Plot a correlation heatmap for the specified columns or entire DataFrame if None.

        :param df: Pandas DataFrame containing the data.
        :param cols: Optional list of column names to include in the correlation.
                     If None, uses all columns in df.
        """
        data = df[cols] if cols else df
        if data.empty:
            logging.error("[DataVisualization] The DataFrame (or chosen subset) is empty. Cannot plot correlation.")
            return
        
        corr_matrix = data.corr(numeric_only=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Correlation Heatmap")

        filename = "correlation_heatmap.png"
        output_path = os.path.join(self.save_dir, filename)
        plt.savefig(output_path)
        plt.close()
        
        logging.info(f"[DataVisualization] Correlation heatmap saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage / Demo
    sample_data = {
        "timestamp": range(50),
        "temp": [25 + (0.1 * i) for i in range(50)],
        "humidity": [60 + (0.05 * i) for i in range(50)]
    }
    df_demo = pd.DataFrame(sample_data)

    viz = DataVisualization(save_dir="demo_visuals")
    
    viz.plot_time_series(
        df_demo,
        time_col="timestamp",
        value_col="temp",
        title="Demo Temperature Over Time",
        x_label="Time (units)",
        y_label="Temperature (°C)"
    )
    
    viz.plot_histogram(df_demo, "humidity", bins=20, kde=True)
    viz.plot_correlation(df_demo, cols=["temp", "humidity"])
    
    logging.info("[DataVisualization] Demo plotting complete.")