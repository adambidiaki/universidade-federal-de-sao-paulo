import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# ==============================
# CONFIGURAÇÕES
# ==============================
BASE_DIR = "imagens"
OUTPUT_DIR = "resultados"
SCALES = list(range(1, 21))
SHAPE = cv2.MORPH_RECT
COLORS = ["#e63946", "#2a9d8f", "#f4a261", "#457b9d", "#6d4c41", "#8e24aa"]


def load_images(base_dir: str) -> dict:
    """
    Lê todas as imagens em subpastas de 'base_dir'.
    """
    classes = {}
    base = Path(base_dir)

    # Cada subpasta representa uma classe
    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir():
            continue

        name = class_dir.name
        images = []

        # Percorre todas as extensões suportadas
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.pgm"):
            for path in sorted(class_dir.glob(ext)):
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

                # Converte para float64 para cálculos de energia
                images.append(img.astype(np.float64))

        # Ignora subpastas vazias
        if images:
            classes[name] = images
            print(f"  Classe '{name}': {len(images)} imagem(ns) carregada(s)")

    return classes


def compute_signatures(classes: dict, scales: list, shape) -> dict:
    """
    Calcula a assinatura granulométrica de cada imagem por classe.
    """
    results = {}

    for class_name, images in classes.items():
        print(f"Processando classe '{class_name}'...")

        class_signatures = []
        for img in images:
            original_energy = img.sum()
            prev_energy = original_energy

            signature = []
            for scale in scales:
                # Elemento estruturante
                kernel = cv2.getStructuringElement(shape, (scale, scale))

                # Abertura morfológica
                opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel).astype(np.float64)
                current_energy = opened.sum()

                # Diferença de energia removida
                diff = prev_energy - current_energy
                signature.append(diff)
                prev_energy = current_energy

            signature = np.array(signature, dtype=np.float64)

            # Normalização pela energia original
            if original_energy > 0:
                signature = signature / original_energy

            class_signatures.append(signature)

        results[class_name] = np.array(class_signatures)

    return results


def plot_individual_signatures(results: dict, scales: list, output_dir: str) -> None:
    """
    Plota todas as assinaturas individuais separadas por classe.
    """
    n_classes = len(results)
    fig, axes = plt.subplots(nrows=1, ncols=n_classes, figsize=(18, 4))

    for ax, (name, signatures), color in zip(axes, results.items(), COLORS):
        for i, signature in enumerate(signatures):
            # Plota cada assinatura individual
            ax.plot(scales, signature, color=color, alpha=0.4, linewidth=1,
                    label="imagens" if i == 0 else "")

        # Média das assinaturas da classe
        ax.plot(scales, signatures.mean(axis=0), color="black", linewidth=2.5,
                linestyle="--", label="media")
        
        # Configurações visuais do subplot
        ax.set_title(f"Classe: {name}", fontsize=12)
        ax.set_xlabel("Raio do elemento estruturante")
        ax.set_ylabel("Energia relativa removida")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Assinaturas Granulométricas", fontsize=14, y=1.02)
    plt.tight_layout()

    # Salva e fecha a figura para liberar memória
    path = os.path.join(output_dir, "assinaturas_individuais.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Salvo: {path}")


def plot_comparison_signatures(results: dict, scales: list, output_dir: str) -> None:
    """
    Plota as curvas médias de todas as classes no mesmo gráfico.
    """
    fig, ax = plt.subplots(figsize=(18, 4))

    for (name, signatures), color in zip(results.items(), COLORS):
        # Média de cada classe em uma única figura
        mean = signatures.mean(axis=0)
        ax.plot(scales, mean, color=color, linewidth=2.5, label=name)

    # Configurações visuais do gráfico
    ax.set_title("Comparação das Assinaturas Granulométricas", fontsize=14)
    ax.set_xlabel("Raio do elemento estruturante")
    ax.set_ylabel("Energia relativa removida")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Salva e fecha a figura para liberar memória
    path = os.path.join(output_dir, "assinaturas_comparacao.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Salvo: {path}")


def export_csv(results: dict, scales: list, output_dir: str) -> None:
    """
    Salva todas as assinaturas em CSV para análise posterior.
    """
    path = os.path.join(output_dir, "assinaturas.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Cabeçalho: identificadores + uma coluna por escala
        header = ["class", "image_idx"] + [f"scale_{scale}" for scale in scales]
        writer.writerow(header)

        # Uma linha por imagem
        for name, signatures in results.items():
            for idx, signature in enumerate(signatures):
                writer.writerow([name, idx] + list(signature))

    print(f"  Salvo: {path}")


def run_analysis() -> None:
    """Executa o fluxo completo de análise granulométrica.

    Etapas:
        1) Carrega imagens organizadas por classe
        2) Calcula assinaturas granulométricas
        3) Gera gráficos individuais e comparativos
        4) Exporta resultados em CSV

    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. Carregando imagens...")
    classes = load_images(BASE_DIR)

    print("\n2. Calculando assinaturas granulométricas...")
    results = compute_signatures(classes, SCALES, SHAPE)

    print("\n3. Gerando gráficos...")
    plot_individual_signatures(results, SCALES, OUTPUT_DIR)
    plot_comparison_signatures(results, SCALES, OUTPUT_DIR)

    print("\n4. Exportando CSV...")
    export_csv(results, SCALES, OUTPUT_DIR)


# ==============================
# EXECUÇÃO
# ==============================
if __name__ == "__main__":
    print("\n--- LABORATÓRIO 03 ---\n")
    run_analysis()