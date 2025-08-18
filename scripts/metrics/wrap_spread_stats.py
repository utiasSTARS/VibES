import argparse
import os
import pickle
import numpy as np
import pandas as pd

def load_all_measures(folder):
    """Load all measures from .pkl files in a folder into a DataFrame."""
    all_data = []
    for file in os.listdir(folder):
        if file.endswith("res.pkl"):
            path = os.path.join(folder, file)
            with open(path, "rb") as f:
                d = pickle.load(f)
                all_data.append(d)  # each d is a dict of measures
    if not all_data:
        raise FileNotFoundError(f"No .pkl files found in {folder}")
    # Convert list of dicts into a DataFrame
    df = pd.DataFrame(all_data)
    return df

def compute_statistics(df):
    """Compute mean, variance, min, max, etc. for all columns."""
    stats = df.agg(['mean', 'var', 'min', 'max', 'median']).transpose()
    return stats

def main():
    parser = argparse.ArgumentParser(description="Compute statistics for event measures from .pkl files")
    parser.add_argument("folder", help="Folder containing .pkl result files")
    parser.add_argument("--output", help="Optional path to save statistics CSV")
    args = parser.parse_args()

    print(f"Loading measures from {args.folder}...")
    df = load_all_measures(args.folder)
    stats = compute_statistics(df)

    print("\nStatistics for each measure:\n")
    print(stats)

    if args.output:
        stats.to_csv(args.output)
        print(f"\nStatistics saved to {args.output}")

if __name__ == "__main__":
    main()
