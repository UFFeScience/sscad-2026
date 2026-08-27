#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import parsl
from parsl.app.app import python_app
from parsl.configs.local_threads import config
from parsl.data_provider.files import File


@python_app
def generate_points_chunk(chunk_id, n_points, dimensions, seed):
    """
    Gera um bloco de pontos sintéticos para o KMeans.
    """
    import random

    rng = random.Random(seed + chunk_id)
    points = []

    for _ in range(n_points):
        cluster_hint = rng.randint(0, 2)
        base = cluster_hint * 10.0
        point = [base + rng.gauss(0, 1) for _ in range(dimensions)]
        points.append(point)

    return points


@python_app
def initialize_centroids(chunks, k):
    """
    Inicializa os centróides usando os primeiros k pontos disponíveis.
    """
    centroids = []

    chunks = [c.result() if hasattr(c, "result") else c for c in chunks]

    for chunk in chunks:
        for point in chunk:
            centroids.append(point)
            if len(centroids) == k:
                return centroids

    raise RuntimeError("Número de pontos menor que k.")


@python_app
def assign_points(points, centroids):
    """
    Associa cada ponto ao centróide mais próximo e calcula somatórios parciais.
    """
    import math

    if hasattr(points, "result"):
        points = points.result()
    if hasattr(centroids, "result"):
        centroids = centroids.result()

    k = len(centroids)
    dimensions = len(centroids[0])

    sums = [[0.0 for _ in range(dimensions)] for _ in range(k)]
    counts = [0 for _ in range(k)]
    sse = 0.0

    for point in points:
        best_idx = None
        best_dist = None

        for i, centroid in enumerate(centroids):
            dist = sum((point[d] - centroid[d]) ** 2 for d in range(dimensions))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i

        counts[best_idx] += 1
        sse += best_dist

        for d in range(dimensions):
            sums[best_idx][d] += point[d]

    return {
        "sums": sums,
        "counts": counts,
        "sse": sse,
    }


@python_app
def update_centroids(partials, old_centroids):
    """
    Reduz os resultados parciais e calcula novos centróides.
    """
    if hasattr(old_centroids, "result"):
        old_centroids = old_centroids.result()
    partials = [p.result() if hasattr(p, "result") else p for p in partials]

    k = len(old_centroids)
    dimensions = len(old_centroids[0])

    total_sums = [[0.0 for _ in range(dimensions)] for _ in range(k)]
    total_counts = [0 for _ in range(k)]

    for partial in partials:
        for i in range(k):
            total_counts[i] += partial["counts"][i]
            for d in range(dimensions):
                total_sums[i][d] += partial["sums"][i][d]

    new_centroids = []

    for i in range(k):
        if total_counts[i] == 0:
            new_centroids.append(old_centroids[i])
        else:
            new_centroids.append([
                total_sums[i][d] / total_counts[i]
                for d in range(dimensions)
            ])

    return new_centroids


@python_app
def write_kmeans_report(final_centroids, final_partials, outputs=[]):
    """
    Escreve um relatório final com os centróides e o SSE.
    """
    output_path = outputs[0].filepath

    if hasattr(final_centroids, "result"):
        final_centroids = final_centroids.result()
    final_partials = [p.result() if hasattr(p, "result") else p for p in final_partials]

    total_sse = sum(partial["sse"] for partial in final_partials)
    total_points = sum(sum(partial["counts"]) for partial in final_partials)

    with open(output_path, "w") as f:
        f.write("KMeans Parsl Workflow\n")
        f.write("=====================\n\n")
        f.write(f"Total de pontos: {total_points}\n")
        f.write(f"SSE final: {total_sse:.6f}\n\n")
        f.write("Centroides finais:\n")

        for i, centroid in enumerate(final_centroids):
            centroid_str = ", ".join(f"{x:.6f}" for x in centroid)
            f.write(f"cluster_{i}: {centroid_str}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Workflow KMeans usando Parsl.")
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/kmeans_report.txt")

    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    parsl.load(config)

    try:
        points_per_chunk = args.points // args.chunks
        remainder = args.points % args.chunks

        chunk_futures = []

        for chunk_id in range(args.chunks):
            n = points_per_chunk + (1 if chunk_id < remainder else 0)
            chunk_futures.append(
                generate_points_chunk(
                    chunk_id,
                    n,
                    args.dimensions,
                    args.seed,
                )
            )

        centroids = initialize_centroids(chunk_futures, args.k)

        for iteration in range(args.iterations):
            partials = [
                assign_points(chunk, centroids)
                for chunk in chunk_futures
            ]

            centroids = update_centroids(partials, centroids)

        final_partials = [
            assign_points(chunk, centroids)
            for chunk in chunk_futures
        ]

        report = write_kmeans_report(
            centroids,
            final_partials,
            outputs=[File(os.path.abspath(args.output))]
        )

        report_path = report.result()

        print(f"Workflow KMeans finalizado.")
        print(f"Relatório: {report_path}")

    finally:
        parsl.clear()


if __name__ == "__main__":
    main()