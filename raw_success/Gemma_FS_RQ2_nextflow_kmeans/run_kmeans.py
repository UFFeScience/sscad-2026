#!/usr/bin/env python3
import argparse
import pandas as pd
from sklearn.cluster import KMeans
import sys

def main():
    parser = argparse.ArgumentParser(description="Perform K-Means Clustering")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to the output results file")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters (default: 3)")

    args = parser.parse_args()

    try:
        # Load data - assumes CSV has no header and contains only numeric values
        data = pd.read_csv(args.input, header=None)
        
        # Initialize and fit KMeans
        kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data)

        # Save labels to output file
        with open(args.output, 'w') as f:
            f.write("Cluster Labels:\n")
            for label in labels:
                f.write(f"{label}\n")
        
        print(f"Successfully clustered data into {args.clusters} groups.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()