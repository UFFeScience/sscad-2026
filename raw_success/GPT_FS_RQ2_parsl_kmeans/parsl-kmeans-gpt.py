#!/usr/bin/env python3

import os
import parsl

from parsl.app.app import python_app
from parsl.configs.local_threads import config

os.makedirs("output", exist_ok=True)

parsl.load(config)


@python_app
def generate_dataset(samples=100000, features=10):
    import numpy as np

    np.random.seed(1234)
    return np.random.rand(samples, features)


@python_app
def run_kmeans(data, n_clusters=8):

    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init="auto"
    )

    model.fit(data)

    return {
        "centers": model.cluster_centers_,
        "labels": model.labels_,
        "inertia": model.inertia_,
    }


@python_app
def save_results(result):

    import numpy as np

    np.save("output/kmeans_centers.npy", result["centers"])
    np.save("output/kmeans_labels.npy", result["labels"])

    with open("output/kmeans_metrics.txt", "w") as f:
        f.write(f"Inertia = {result['inertia']}\n")

    return "output/kmeans_metrics.txt"


if __name__ == "__main__":

    dataset = generate_dataset()

    clustering = run_kmeans(dataset)

    output = save_results(clustering)

    print(output.result())

    parsl.clear()
