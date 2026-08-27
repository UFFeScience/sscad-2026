import numpy as np
import time

from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier
from pycompss.api.parameter import IN, OUT


# ============================================================
# KMEANS WORKFLOW
# ============================================================

@task(returns=np.ndarray)
def compute_distances(points, centroid):
    """
    Calcula distância dos pontos para um centróide.
    """
    distances = np.linalg.norm(points - centroid, axis=1)
    return distances


@task(returns=np.ndarray)
def assign_clusters(distance_blocks):
    """
    Define cluster de menor distância.
    """
    return np.argmin(distance_blocks, axis=0)



@task(returns=np.ndarray)
def update_centroid(points, labels, cluster_id):
    """
    Atualiza centróide.
    """
    cluster_points = points[labels == cluster_id]

    if len(cluster_points) == 0:
        return np.zeros(points.shape[1])

    return np.mean(cluster_points, axis=0)



def kmeans_workflow(
        data,
        k=3,
        iterations=5):

    print("\nExecutando KMeans")

    centroids = data[:k]

    for _ in range(iterations):

        distance_tasks = []

        for centroid in centroids:
            task_result = compute_distances(
                data,
                centroid
            )

            distance_tasks.append(task_result)


        distances = compss_wait_on(distance_tasks)

        distances = np.array(distances)


        labels = assign_clusters(
            distances
        )


        labels = compss_wait_on(labels)


        centroid_tasks = []

        for cluster in range(k):

            centroid_tasks.append(
                update_centroid(
                    data,
                    labels,
                    cluster
                )
            )


        centroids = compss_wait_on(
            centroid_tasks
        )


    print("KMeans finalizado")

    return centroids



# ============================================================
# MAIN WORKFLOW
# ============================================================


def main():

    # -------------------------------
    # KMeans
    # -------------------------------

    data = np.random.rand(
        10000,
        10
    )


    centroids = kmeans_workflow(
        data,
        k=5
    )


    print(
        "Centroids:",
        centroids
    )


    compss_barrier()


if __name__ == "__main__":

    main()
