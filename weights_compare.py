import os
import json
import glob
import re
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Important metrics to track
important_keys = [
    "position_error",
    "rotation_error",
    "solve_time",
    "total_time",
    "optimized_dt",
    "trajopt_attempts"
]

# Regex to extract datetime from filename
FILENAME_PATTERN = re.compile(r"results_(\d{8}_\d{6})\.json")

def extract_timestamp(filename):
    match = FILENAME_PATTERN.search(filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return None

def load_results(log_dir="logs"):
    results = []

    for filepath in sorted(glob.glob(os.path.join(log_dir, "results_*.json"))):
        with open(filepath, 'r') as f:
            data = json.load(f)

        timestamp = extract_timestamp(filepath)

        for segment in ["home_to_pick", "pick_to_place"]:
            segment_data = data.get(segment, {})
            row = {
                "timestamp": timestamp,
                "filename": os.path.basename(filepath),
                "segment": segment
            }
            for key in important_keys:
                row[key] = segment_data.get(key)
            results.append(row)

    return pd.DataFrame(results)

def plot_all_metrics(df):
    # Ensure chronological order
    df = df.sort_values(by="timestamp")

    # Create one figure per segment
    for segment in df['segment'].unique():
        segment_df = df[df['segment'] == segment]
        fig, axs = plt.subplots(len(important_keys), 1, figsize=(12, 2.8 * len(important_keys)), sharex=True)

        fig.suptitle(f"MotionGenResult Metrics - Segment: {segment}", fontsize=16, y=1.02)

        for i, metric in enumerate(important_keys):
            axs[i].plot(segment_df['timestamp'], segment_df[metric], marker='o', linestyle='-')
            axs[i].set_ylabel(metric)
            axs[i].grid(True)
            axs[i].set_title(metric)

        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        axs[-1].set_xlabel("Timestamp")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f'results_{segment}.png')

if __name__ == "__main__":
    df = load_results()
    if df.empty:
        print("No result files found.")
    else:
        print(df.head())
        plot_all_metrics(df)
