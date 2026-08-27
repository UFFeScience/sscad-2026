#!/usr/bin/env python3
"""
matmul.py
Workflow Parsl para calcular a multiplicação de matrizes (Matmul).
Gera matrizes aleatórias, executa a multiplicação e salva a soma dos elementos.
"""

import os
import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config
import numpy as np

# ---------------------------------------------------------------------
# Configuração e inicialização
# ---------------------------------------------------------------------
os.makedirs('output', exist_ok=True)

def main():
    # Carregar a configuração do Parsl
    parsl.load(config)

    # ---------------------------------------------------------------------
    # Definir a aplicação Python para Matmul
    # ---------------------------------------------------------------------
    @python_app
    def run_matmul():
        """
        Gera matrizes aleatórias e executa a multiplicação de matrizes.
        """
        # 1. Gerar matrizes aleatórias
        np.random.seed(123)
        A = np.random.randint(1, 10, size=(5, 5))
        B = np.random.randint(1, 10, size=(5, 5))

        # 2. Calcular Matmul
        C = np.dot(A, B)

        # 3. Calcular um valor de verificação simples (traço)
        trace = np.trace(C)
        sum_elements = np.sum(C)

        # 4. Salvar resultados
        output_path = 'output/matmul_results.txt'
        with open(output_path, 'w') as f:
            f.write("Matriz A:\n")
            f.write(str(A) + "\n\n")
            f.write("Matriz B:\n")
            f.write(str(B) + "\n\n")
            f.write(f"Traço da Matriz Resultante (C): {trace}\n")
            f.write(f"Soma de todos os elementos de C: {sum_elements}\n")

        print(f"Matmul concluído. Resultados salvos em {output_path}")
        return output_path

    # ---------------------------------------------------------------------
    # Executar o workflow
    # ---------------------------------------------------------------------
    result_future = run_matmul()
    output_file = result_future.result()

    # Imprimir conteúdo do arquivo
    with open(output_file, 'r') as f:
        print("\n--- Conteúdo do Arquivo de Resultados ---")
        print(f.read())

    parsl.clear()

if __name__ == "__main__":
    main()