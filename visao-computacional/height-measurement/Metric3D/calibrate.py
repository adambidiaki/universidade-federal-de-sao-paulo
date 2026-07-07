"""
calibrate.py
------------
Seleciona uma imagem, roda inferência do Metric3D V2, e permite ao usuário
clicar em um objeto de altura conhecida para computar o fator de escala.

Fluxo de cliques (janela "RGB"):
  1. Clique no objeto de referência  (qualquer ponto — contexto visual)
  2. Clique na BASE do objeto
  3. Clique no TOPO do objeto
  → Terminal pede a altura real → salva calibration.json

A janela "Depth" é somente para observação da profundidade.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tkinter import Tk, filedialog

# ==============================
# CONFIGURAÇÕES
# ==============================
# Parâmetros intrínsecos da câmera
FX = 718.856
FY = 718.856
CX = 607.1928
CY = 185.2157

CALIB_FILE = "calibration.json"

# Paleta de cores dos marcadores e anotações
COLOR_OBJ = (0, 255, 255)
COLOR_BASE = (0, 255, 0)
COLOR_TOP = (0, 128, 255)
COLOR_LINE = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==============================
# ESTADO GLOBAL
# ==============================
state = 0  # etapa atual da calibração (0=ref, 1=base, 2=topo, 3=concluído)
points = {}  # pontos clicados na calibração em andamento
depth = None  # mapa de profundidade inferido
rgb_bgr = None  # cópia da imagem para renderização
w0 = h0 = 0  # dimensões da imagem

# Instruções e dicas de estado exibidas na janela RGB
INSTRUCTIONS = [
    ("1. Clique no objeto de referencia (qualquer ponto)", COLOR_OBJ),
    ("2. Clique na BASE do objeto",                        COLOR_BASE),
    ("3. Clique no TOPO do objeto",                        COLOR_TOP),
]

STATE_HINTS = [
    (">> Aguardando: ponto de referencia", COLOR_OBJ),
    (">> Aguardando: BASE do objeto",      COLOR_BASE),
    (">> Aguardando: TOPO do objeto",      COLOR_TOP),
]


# ==============================
# UTILITÁRIOS VISUAIS
# ==============================
def draw_marker(
    img: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
    label: str
) -> None:
    """Desenha um marcador circular com rótulo de texto na imagem.

    Args:
        img (np.ndarray): Imagem onde o marcador será desenhado.
        x (int): Coordenada X do centro do marcador.
        y (int): Coordenada Y do centro do marcador.
        color (tuple[int, int, int]): Cor do marcador em BGR.
        label (str): Texto a exibir ao lado do marcador.

    """
    # Círculo preenchido com borda branca
    cv2.circle(img, (x, y), 7, color, -1)
    cv2.circle(img, (x, y), 9, (255, 255, 255), 1)

    # Texto com sombra preta
    cv2.putText(img, label, (x + 12, y + 5), FONT, 0.55, (255, 255, 255), 2)
    cv2.putText(img, label, (x + 12, y + 5), FONT, 0.55, color, 1)


def draw_hud(
    img: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    start_y: int = 20
) -> None:
    """Renderiza linhas de texto sobrepostas na imagem (HUD).

    Args:
        img (np.ndarray): Imagem onde o HUD será desenhado.
        lines (list[tuple[str, tuple]]): Lista de pares (texto, cor BGR).
        start_y (int): Posição Y inicial do primeiro texto.

    """
    for i, (text, color) in enumerate(lines):
        y = start_y + i * 22

        # Sombra preta seguida do texto colorido
        cv2.putText(img, text, (10, y), FONT, 0.55, (0, 0, 0), 3)
        cv2.putText(img, text, (10, y), FONT, 0.55, color, 1)


def to_3d(
    x: int,
    y: int,
    z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> np.ndarray:
    """Converte coordenadas de pixel e profundidade para ponto 3D.

    Args:
        x (int): Coordenada X do pixel.
        y (int): Coordenada Y do pixel.
        z (float): Profundidade em metros.
        fx (float): Distância focal horizontal.
        fy (float): Distância focal vertical.
        cx (float): Centro óptico horizontal.
        cy (float): Centro óptico vertical.

    Returns:
        np.ndarray: Ponto 3D [X, Y, Z] em metros.

    """
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy

    return np.array([X, Y, z])


def depth_colormap(depth: np.ndarray) -> np.ndarray:
    """Converte mapa de profundidade em imagem colorida para visualização.

    Args:
        depth (np.ndarray): Mapa de profundidade em metros.

    Returns:
        np.ndarray: Imagem BGR com colormap aplicado.

    """
    # Normaliza entre 0 e 255 para o colormap
    norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    gray = (norm * 255).astype(np.uint8)

    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


# ==============================
# INTERFACE DE CALIBRAÇÃO
# ==============================
def refresh_rgb_window() -> None:
    """Redesenha a janela RGB com marcadores, HUD e dica de estado atualizados."""

    canvas = rgb_bgr.copy()

    # Instruções fixas no topo
    draw_hud(canvas, INSTRUCTIONS, start_y=20)

    # Dica dinâmica da etapa atual na parte inferior
    if state < 3:
        hint_text, hint_color = STATE_HINTS[state]
        cv2.putText(canvas, hint_text, (10, h0 - 12), FONT, 0.55, (0, 0, 0), 3)
        cv2.putText(canvas, hint_text, (10, h0 - 12), FONT, 0.55, hint_color, 1)

    # Marcadores dos pontos já clicados
    if "obj" in points:
        x, y, z = points["obj"]
        draw_marker(canvas, x, y, COLOR_OBJ, f"REF z={z:.2f}m")

    if "base" in points:
        x, y, z = points["base"]
        draw_marker(canvas, x, y, COLOR_BASE, f"BASE z={z:.2f}m")

    if "top" in points:
        x, y, z = points["top"]
        draw_marker(canvas, x, y, COLOR_TOP, f"TOP z={z:.2f}m")

    # Linha parcial quando base e topo estão definidos
    if "base" in points and "top" in points:
        bx, by, _ = points["base"]
        tx, ty, _ = points["top"]
        cv2.line(canvas, (bx, by), (tx, ty), COLOR_LINE, 2)

    cv2.imshow("RGB — Calibracao", canvas)


def ask_real_height() -> float:
    """Solicita ao usuário a altura real do objeto de referência via terminal.

    Repete a pergunta até que um valor positivo válido seja informado.

    Returns:
        float: Altura real do objeto em metros.

    """
    real_height = None

    while real_height is None:
        try:
            val = float(input("Digite a altura REAL do objeto (em metros): "))
            if val <= 0:
                raise ValueError
            real_height = val
        except ValueError:
            print("  Valor inválido. Use um número positivo (ex: 1.75)")

    return real_height


def save_calibration(
    scale: float,
    raw_height: float,
    real_height: float
) -> None:
    """Salva os parâmetros de calibração em um arquivo JSON.

    Args:
        scale (float): Fator de escala calculado (real / bruto).
        raw_height (float): Altura estimada pelo modelo sem correção.
        real_height (float): Altura real informada pelo usuário.

    """
    calib = {
        "scale": scale,
        "fx": FX, "fy": FY, "cx": CX, "cy": CY,
        "raw_height": raw_height,
        "real_height": real_height,
        "points": {
            "obj": list(points["obj"]),
            "base": list(points["base"]),
            "top": list(points["top"]),
        },
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)

    print("\n========================================")
    print(f"  Altura bruta  : {raw_height:.4f} m")
    print(f"  Altura real   : {real_height:.4f} m")
    print(f"  Escala        : {scale:.6f}")
    print(f"  Salvo em      : {os.path.abspath(CALIB_FILE)}")
    print("========================================\n")
    print("Feche as janelas para sair.")


def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
    """Callback do mouse para coleta dos três pontos de calibração.

    Segue um ciclo de 3 cliques: referência → base → topo.
    Ao completar o ciclo, solicita a altura real e salva a calibração.

    Args:
        event (int): Tipo do evento do mouse.
        x (int): Coordenada X do cursor na imagem exibida.
        y (int): Coordenada Y do cursor na imagem exibida.
        flags (int): Flags adicionais do evento (não utilizado).
        param (object): Parâmetro extra do callback (não utilizado).

    """
    global state, points

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Ignora cliques fora dos limites da imagem
    if x < 0 or y < 0 or x >= w0 or y >= h0:
        return

    z = float(depth[y, x])

    if state == 0:
        # Ponto de referência visual do objeto
        points["obj"] = (x, y, z)
        state = 1
        print(f"[REF]  ({x:4d},{y:4d})  depth={z:.3f} m")
        print("→ Clique na BASE do objeto")

    elif state == 1:
        # Ponto da base do objeto
        points["base"] = (x, y, z)
        state = 2
        print(f"[BASE] ({x:4d},{y:4d})  depth={z:.3f} m")
        print("→ Clique no TOPO do objeto")

    elif state == 2:
        # Ponto do topo: calcula escala e salva calibração
        points["top"] = (x, y, z)

        # Usa profundidade média entre base e topo para projeção 3D
        z_ref = (points["base"][2] + points["top"][2]) / 2.0
        p1 = to_3d(points["base"][0], points["base"][1], z_ref, FX, FY, CX, CY)
        p2 = to_3d(points["top"][0],  points["top"][1],  z_ref, FX, FY, CX, CY)
        raw_height = float(np.linalg.norm(p2 - p1))

        print(f"[TOP]  ({x:4d},{y:4d})  depth={z:.3f} m")
        print(f"\n[INFO] Altura bruta (sem escala): {raw_height:.4f} m")

        # Solicita altura real e computa fator de escala
        real_height = ask_real_height()
        scale = real_height / raw_height

        save_calibration(scale, raw_height, real_height)

        state = 3

    refresh_rgb_window()


def main() -> None:
    """Executa o fluxo completo de calibração.

    Etapas:
        1) Seleciona imagem via diálogo
        2) Infere mapa de profundidade com Metric3D V2
        3) Abre janelas de interação (RGB + Depth)
        4) Aguarda os três cliques de calibração
        5) Calcula escala e salva calibration.json

    """
    global depth, rgb_bgr, w0, h0

    # Seleciona imagem via diálogo de arquivo
    Tk().withdraw()
    img_path = filedialog.askopenfilename(
        title="Selecione a imagem para calibração",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
    )
    if not img_path:
        print("Nenhuma imagem selecionada. Encerrando.")
        sys.exit(0)

    # Carrega imagem e prepara cópia para renderização
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Erro ao abrir: {img_path}")
        sys.exit(1)

    rgb_bgr = bgr.copy()
    h0, w0  = bgr.shape[:2]

    # Pré-processa imagem para inferência
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Carrega modelo de profundidade monocular
    print("Carregando modelo Metric3D V2 …")
    model = torch.hub.load(
        ".", "metric3d_vit_small", source="local", pretrain=True
    ).cuda()
    model.eval()

    # Redimensiona para resolução esperada pelo modelo e infere profundidade
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).cuda()
    img_t = F.interpolate(img_t, size=(392, 1280), mode="bilinear", align_corners=False)

    print("Rodando inferência …")
    with torch.no_grad():
        result = model.inference({"input": img_t})

    # Redimensiona mapa de profundidade para o tamanho original da imagem
    raw_depth = result[0][0, 0].detach().cpu().numpy()
    depth = cv2.resize(raw_depth, (w0, h0))
    depth_display = depth_colormap(depth)

    # Configura janelas lado a lado
    cv2.namedWindow("RGB — Calibracao",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth — Calibracao", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB — Calibracao",   w0, h0)
    cv2.resizeWindow("Depth — Calibracao", w0, h0)
    cv2.moveWindow("RGB — Calibracao",     0,       50)
    cv2.moveWindow("Depth — Calibracao",   w0 + 10, 50)

    cv2.imshow("Depth — Calibracao", depth_display)
    cv2.setMouseCallback("RGB — Calibracao", mouse_callback)

    print("\n== CALIBRAÇÃO ==")
    print("→ Clique em qualquer ponto do objeto de referência na janela RGB")

    refresh_rgb_window()

    # Loop principal: aguarda conclusão ou fechamento da janela
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        if state == 3 and key in (13, ord('q')):
            break
        if cv2.getWindowProperty("RGB — Calibracao", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

# ==============================
# EXECUÇÃO
# ==============================
if __name__ == "__main__":
    print("\n--- CALIBRAÇÃO DE ALTURA ---\n")
    main()