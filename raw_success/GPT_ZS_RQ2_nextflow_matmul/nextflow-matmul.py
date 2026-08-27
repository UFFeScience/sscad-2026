import numpy as np
import time
import sys



size = int(sys.argv[1])


print("Executando Matmul")



A = np.random.rand(size,size)

B = np.random.rand(size,size)



start = time.time()



C = np.matmul(A,B)



elapsed = time.time()-start



with open("matmul_result.txt","w") as f:

    f.write(
        f"Matrix Multiplication\n"
        f"Tamanho: {size}x{size}\n"
        f"Tempo: {elapsed}s\n"
    )



print("Matmul finalizado")