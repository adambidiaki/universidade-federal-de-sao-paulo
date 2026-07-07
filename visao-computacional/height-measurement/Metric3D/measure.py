"""
measure.py
----------
Carrega calibration.json e permite medir a altura de objetos arbitrários
em uma imagem estática usando o fator de escala salvo.

Fluxo de cliques (janela "RGB"):
  1. Clique no objeto a medir   (contexto visual)
  2. Clique na BASE do objeto
  3. Clique no TOPO do objeto
  → Exibe altura estimada na imagem e no terminal
  → Clique novamente para medir outro objeto

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
CALIB_FILE = "model/calibration.json"

# Configurações do zoom
SCALE = 1
ZOOM_HALF_SIZE = 80  # raio do recorte na imagem original (px)
ZOOM_WIN_SIZE = 300  # tamanho da janela de zoom (px)

# Paleta de cores dos marcadores e anotações
COLOR_OBJ = (0, 255, 255)
COLOR_BASE = (0, 255, 0)
COLOR_TOP = (0, 128, 255)
COLOR_LINE = (255, 0, 255)
COLOR_RESULT = (255, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==============================
# ESTADO GLOBAL
# ==============================
state = 0  # etapa atual da medição (0=ref, 1=base, 2=topo)
points = {}  # pontos da medição em andamento
measurements = []  # medições finalizadas

depth = None  # mapa de profundidade inferido
bgr = None  # imagem original em BGR
rgb_bgr = None  # cópia da imagem para renderização
w0 = h0 = 0  # dimensões da imagem

scale = fx = fy = cx = cy = None  # parâmetros de calibração

# Instruções e dicas de estado exibidas na janela RGB
INSTRUCTIONS = [
    ("1. Clique no objeto a medir (qualquer ponto)", COLOR_OBJ),
    ("2. Clique na BASE do objeto", COLOR_BASE),
    ("3. Clique no TOPO do objeto", COLOR_TOP),
]

STATE_HINTS = [
    (">> Aguardando: ponto de referencia", COLOR_OBJ),
    (">> Aguardando: BASE do objeto", COLOR_BASE),
    (">> Aguardando: TOPO do objeto", COLOR_TOP),
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


def show_zoom(x_display: int, y_display: int) -> None:
    """Atualiza a janela de zoom com base na posição do mouse na imagem exibida.

    Mantém fator de zoom constante. Regiões fora da imagem são preenchidas com preto.

    Args:
        x_display (int): Coordenada X do mouse na imagem redimensionada.
        y_display (int): Coordenada Y do mouse na imagem redimensionada.

    """
    original_img = bgr.copy()

    # Converte coordenadas da imagem redimensionada para a imagem original
    x_original = int(x_display / SCALE)
    y_original = int(y_display / SCALE)

    height, width = original_img.shape[:2]
    zoom_size = 2 * ZOOM_HALF_SIZE

    # Coordenadas desejadas do recorte na imagem original
    x1, x2 = x_original - ZOOM_HALF_SIZE, x_original + ZOOM_HALF_SIZE
    y1, y2 = y_original - ZOOM_HALF_SIZE, y_original + ZOOM_HALF_SIZE

    # Background preto como base para o recorte
    background = np.zeros((zoom_size, zoom_size, 3), dtype=original_img.dtype)

    # Interseção do recorte com os limites reais da imagem
    clip_x1, clip_y1 = max(0, x1), max(0, y1)
    clip_x2, clip_y2 = min(width, x2), min(height, y2)

    # Sem interseção: nada a exibir
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return

    # Recorte real da imagem e posição de destino no background
    roi = original_img[clip_y1:clip_y2, clip_x1:clip_x2]
    offset_x, offset_y = clip_x1 - x1, clip_y1 - y1

    # Cola o recorte no background na posição correta
    roi_height, roi_width = roi.shape[:2]
    background[offset_y:offset_y + roi_height, offset_x:offset_x + roi_width] = roi

    # Amplia o background para o tamanho da janela de zoom
    zoom = cv2.resize(background, (ZOOM_WIN_SIZE, ZOOM_WIN_SIZE), interpolation=cv2.INTER_LINEAR)

    # Cruz no centro para indicar o pixel apontado
    zoom_center = ZOOM_WIN_SIZE // 2
    cv2.line(zoom, (zoom_center, 0), (zoom_center, ZOOM_WIN_SIZE - 1), (0, 255, 255), 1)
    cv2.line(zoom, (0, zoom_center), (ZOOM_WIN_SIZE - 1, zoom_center), (0, 255, 255), 1)

    cv2.imshow("Zoom", zoom)


# ==============================
# INTERFACE DE MEDIÇÃO
# ==============================
def refresh_rgb_window() -> None:
    """Redesenha a janela RGB com marcadores, medições e HUD atualizados."""

    canvas = rgb_bgr.copy()

    # Instruções fixas no topo
    draw_hud(canvas, INSTRUCTIONS, start_y=20)

    # Dica dinâmica da etapa atual na parte inferior
    if state < 3:
        hint_text, hint_color = STATE_HINTS[state]
        cv2.putText(canvas, hint_text, (10, h0 - 12), FONT, 0.55, (0, 0, 0), 3)
        cv2.putText(canvas, hint_text, (10, h0 - 12), FONT, 0.55, hint_color, 1)

    # Renderiza medições já finalizadas
    for m in measurements:
        bx, by, _ = m["base"]
        tx, ty, _ = m["top"]

        cv2.line(canvas, (bx, by), (tx, ty), COLOR_LINE, 2)
        cv2.circle(canvas, (bx, by), 6, COLOR_BASE, -1)
        cv2.circle(canvas, (tx, ty), 6, COLOR_TOP,  -1)

        # Rótulo com altura no ponto médio da linha
        mid_x = (bx + tx) // 2
        mid_y = (by + ty) // 2
        label = f"{m['height']:.2f} m"
        cv2.putText(canvas, label, (mid_x + 8, mid_y), FONT, 0.7, (0, 0, 0), 4)
        cv2.putText(canvas, label, (mid_x + 8, mid_y), FONT, 0.7, COLOR_RESULT, 2)

    # Marcadores da medição em andamento
    if "obj" in points:
        x, y, z = points["obj"]
        draw_marker(canvas, x, y, COLOR_OBJ, f"REF z={z:.2f}m")

    if "base" in points:
        x, y, z = points["base"]
        draw_marker(canvas, x, y, COLOR_BASE, f"BASE z={z:.2f}m")

    if "top" in points:
        x, y, z = points["top"]
        draw_marker(canvas, x, y, COLOR_TOP, f"TOP z={z:.2f}m")

    # Linha parcial quando base já foi definida
    if "base" in points and "top" in points:
        bx, by, _ = points["base"]
        tx, ty, _ = points["top"]
        cv2.line(canvas, (bx, by), (tx, ty), COLOR_LINE, 2)

    cv2.imshow("RGB — Medicao", canvas)


def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
    """Callback do mouse para coleta de pontos e cálculo de altura.

    Segue um ciclo de 3 cliques: referência → base → topo.
    Ao completar o ciclo, calcula e registra a altura estimada.

    Args:
        event (int): Tipo do evento do mouse.
        x (int): Coordenada X do cursor na imagem exibida.
        y (int): Coordenada Y do cursor na imagem exibida.
        flags (int): Flags adicionais do evento (não utilizado).
        param (object): Parâmetro extra do callback (não utilizado).

    """
    global state, points

    # Atualiza janela de zoom a cada movimento
    if event == cv2.EVENT_MOUSEMOVE:
        show_zoom(x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Ignora cliques fora dos limites da imagem
    if x < 0 or y < 0 or x >= w0 or y >= h0:
        return

    z = float(depth[y, x])

    if state == 0:
        # Ponto de referência: reinicia medição anterior
        points = {}
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
        # Ponto do topo: calcula e registra a altura
        points["top"] = (x, y, z)

        # Usa profundidade média entre base e topo para projeção 3D
        z_ref = (points["base"][2] + points["top"][2]) / 2.0
        p1 = to_3d(points["base"][0], points["base"][1], z_ref, fx, fy, cx, cy)
        p2 = to_3d(points["top"][0],  points["top"][1],  z_ref, fx, fy, cx, cy)

        raw_height       = float(np.linalg.norm(p2 - p1))
        corrected_height = raw_height * scale

        measurements.append({
            "base": points["base"],
            "top": points["top"],
            "height": corrected_height,
        })

        print(f"[TOP]  ({x:4d},{y:4d})  depth={z:.3f} m")
        print("\n========================================")
        print(f"  Altura bruta  : {raw_height:.4f} m  (sem escala)")
        print(f"  Escala        : {scale:.6f}")
        print(f"  Altura final  : {corrected_height:.4f} m")
        print("========================================")
        print("Clique novamente para medir outro objeto.\n")

        state = 0

    refresh_rgb_window()


def main() -> None:
    """Executa o fluxo completo de medição.

    Etapas:
        1) Carrega calibração salva
        2) Seleciona imagem via diálogo
        3) Infere mapa de profundidade com Metric3D V2
        4) Abre janelas de interação (RGB + Depth + Zoom)
        5) Aguarda medições até ESC ou Q

    """
    global depth, bgr, rgb_bgr, w0, h0
    global scale, fx, fy, cx, cy

    # Valida existência do arquivo de calibração
    if not os.path.exists(CALIB_FILE):
        print(f"Arquivo '{CALIB_FILE}' não encontrado.")
        print("Execute 'calibrate.py' primeiro.")
        sys.exit(1)

    # Carrega parâmetros de calibração
    with open(CALIB_FILE) as f:
        calib = json.load(f)

    scale = calib["scale"]
    fx = calib["fx"]
    fy = calib["fy"]
    cx = calib["cx"]
    cy = calib["cy"]

    print(f"Calibração carregada: scale={scale:.6f}")

    # Seleciona imagem via diálogo de arquivo
    Tk().withdraw()
    img_path = filedialog.askopenfilename(
        title="Selecione a imagem para medição",
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
    h0, w0 = bgr.shape[:2]

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
    cv2.namedWindow("RGB — Medicao", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth — Medicao", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB — Medicao", w0, h0)
    cv2.resizeWindow("Depth — Medicao", w0, h0)
    cv2.moveWindow("RGB — Medicao", 0, 50)
    cv2.moveWindow("Depth — Medicao", w0 + 10, 50)

    cv2.imshow("Depth — Medicao", depth_display)
    cv2.setMouseCallback("RGB — Medicao", mouse_callback)

    print("\n== MEDIÇÃO ==")
    print("Clique na janela RGB seguindo as instruções.")
    print("ESC ou Q para sair.\n")

    refresh_rgb_window()

    # Loop principal: aguarda interação até fechamento ou tecla de saída
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break
        if (cv2.getWindowProperty("RGB — Medicao", cv2.WND_PROP_VISIBLE) < 1 or
                cv2.getWindowProperty("Depth — Medicao", cv2.WND_PROP_VISIBLE) < 1):
            break

    cv2.destroyAllWindows()

# ==============================
# EXECUÇÃO
# ==============================
if __name__ == "__main__":
    print("\n--- MEDIÇÃO DE ALTURA ---\n")
    main()