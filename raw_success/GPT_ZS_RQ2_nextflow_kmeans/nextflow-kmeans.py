import numpy as np
import time
import sys

from sklearn.cluster import KMeans



size = int(sys.argv[1])


print("Executando KMeans")


# geração dos dados

data = np.random.rand(size, 3)



start = time.time()



model = KMeans(
    n_clusters=5,
    random_state=0,
    n_init=10
)


model.fit(data)



elapsed = time.time() - start



with open("kmeans_result.txt","w") as f:

    f.write(
        f"KMeans executado\n"
        f"Pontos: {size}\n"
        f"Tempo: {elapsed}s\n"
    )



print("KMeans finalizado")