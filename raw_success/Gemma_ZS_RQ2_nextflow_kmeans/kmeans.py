import sys
import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans(csv_file):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Simple KMeans with 2 clusters
    kmeans = KMeans(n_clusters=2, n_init=10).fit(df)
    
    # Print labels to stdout
    print("Cluster Labels:")
    print(kmeans.labels_)

if __name__ == "__main__":
    run_kmeans(sys.argv[1])