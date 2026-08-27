import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# Configuração do Parsl
parsl.load(config)


# Tarefa distribuída
@python_app
def kmeans_task(data, n_clusters):
    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=n_clusters,
        random_state=0
    )

    labels = model.fit_predict(data)

    return labels.tolist()



def kmeans_workflow():

    # Geração dos dados
    data = make_blobs(
        n_samples=10000,
        centers=5,
        n_features=10,
        random_state=42
    )[0]


    # Chamada assíncrona
    result = kmeans_task(
        data,
        5
    )


    # Espera resultado
    labels = result.result()


    print(
        "KMeans finalizado"
    )

    print(
        "Clusters encontrados:",
        np.unique(labels)
    )



if __name__ == "__main__":

    kmeans_workflow()