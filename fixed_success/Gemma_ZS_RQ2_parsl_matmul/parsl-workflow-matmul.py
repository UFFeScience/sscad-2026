import parsl



from parsl.app.app import python_app



from parsl.configs.local_threads import config as local_config



parsl.load(local_config)



@python_app
def run_matmul(size=1024):
    """Simula a multiplicação de matrizes."""
    import numpy as np
    print(f"Iniciando MatMul de tamanho {size}x{size}...")
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.dot(A, B)
    return f"MatMul concluído. Shape do resultado: {C.shape}"



def main():
    print("Iniciando Workflow Paralelo via Parsl...\n")

    future_matmul = run_matmul(2048)

    # Coletando resultados
    print("Aguardando finalização das tarefas...")
    print(f"Resultado MatMul: {future_matmul.result()}")



if __name__ == "__main__":
    main()
