import cv2
import numpy as np
import math
import json
from tkinter import Tk, filedialog

# ==============================
# CONFIGURAÇÕES
# ==============================
MODEL_PATH = "models/height_model.json"

# Configurações do zoom
SCALE = 0.7
ZOOM_HALF_SIZE = 80
ZOOM_WIN_SIZE = 300

clicked_points = []
original_img = None


def select_image() -> str:
    """Abre um diálogo para o usuário selecionar uma imagem.

    Returns:
        str: Caminho da imagem selecionada.

    Raises:
        RuntimeError: Caso nenhuma imagem seja selecionada.

    """
    root = Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Selecione a imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )

    if not path:
        raise RuntimeError("Nenhuma imagem selecionada.")

    return path


def resize_for_display(img: np.ndarray,  scale: float) -> np.ndarray:
    """Redimensiona imagem mantendo proporção para exibição.

    Args:
        img (np.ndarray): Imagem original.
        scale (float): Fator de escala.

    Returns:
        np.ndarray: Imagem redimensionada.

    """
    height, width = img.shape[:2]
    new_size = (int(width * scale), int(height * scale))

    resized = cv2.resize(img, new_size)

    return resized


def distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Calcula distância Euclidiana entre dois pontos.

    Args:
        p1 (tuple[int, int]): Primeiro ponto.
        p2 (tuple[int, int]): Segundo ponto.

    Returns:
        float: Distância.

    """
    distance = math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    return distance


def load_model(path: str) -> dict:
    """Carrega modelo de calibração salvo.

    Args:
        path (str): Caminho do JSON.

    Returns:
        dict: Estrutura do modelo.

    """
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    return model


def estimate_height(
    model: dict,
    base: tuple[int, int],
    top: tuple[int, int]
) -> float:
    """Estima altura de um objeto usando modelo calibrado.

    Args:
        model (dict): Modelo carregado.
        base (tuple[int, int]): Base do objeto.
        top (tuple[int, int]): Topo do objeto.

    Returns:
        float: Altura estimada (m).

    """
    vp = model["vp"]
    base_ref = model["base_ref"]
    top_ref = model["top_ref"]
    real_height = model["real_height"]

    height_ref = distance(base_ref, top_ref)
    height_obj = distance(base, top)

    distance_ref = distance(vp, base_ref)
    distance_obj = distance(vp, base)

    if distance_obj == 0:
        return 0

    height = real_height * (height_obj / height_ref) * (distance_ref / distance_obj)

    return height


def show_zoom(x_display: int, y_display: int) -> None:
    """Atualiza a janela de zoom com base na posição do mouse na imagem exibida, mantendo sempre o mesmo 
    fator de zoom. Quando o mouse chega perto da borda, a parte que sair da imagem é preenchida com preto.

    Args:
        x_display (int): Coordenada X do mouse na imagem redimensionada.
        y_display (int): Coordenada Y do mouse na imagem redimensionada.

    """
    # Converte coordenadas da imagem redimensionada para a imagem original
    x_original = int(x_display / SCALE)
    y_original = int(y_display / SCALE)

    height, width = original_img.shape[:2]
    zoom_size = 2 * ZOOM_HALF_SIZE

    # Coordenadas desejadas do recorte na imagem original
    x1 = x_original - ZOOM_HALF_SIZE
    x2 = x_original + ZOOM_HALF_SIZE
    y1 = y_original - ZOOM_HALF_SIZE
    y2 = y_original + ZOOM_HALF_SIZE

    # Cria um "background" preto do tamanho fixo do recorte
    background = np.zeros((zoom_size, zoom_size, 3), dtype=original_img.dtype)

    # Interseção desse recorte com a imagem (parte que realmente existe)
    clip_x1 = max(0, x1)
    clip_y1 = max(0, y1)
    clip_x2 = min(width, x2)
    clip_y2 = min(height, y2)

    # Se não tiver interseção, não faz nada
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return

    # Recorte real da imagem
    roi = original_img[clip_y1:clip_y2, clip_x1:clip_x2]

    # Posição onde esse recorte entra dentro do background
    offset_x = clip_x1 - x1
    offset_y = clip_y1 - y1

    # Colar o ROI no background na posição certa
    roi_height, roi_width = roi.shape[:2]
    background[offset_y:offset_y + roi_height, offset_x:offset_x + roi_width] = roi

    # Redimensiona o background para o tamanho da janela de zoom
    zoom = cv2.resize(background, (ZOOM_WIN_SIZE, ZOOM_WIN_SIZE), interpolation=cv2.INTER_LINEAR)

    # Cruz no centro da janela de zoom
    zoom_center = ZOOM_WIN_SIZE // 2
    cv2.line(zoom, (zoom_center, 0), (zoom_center, ZOOM_WIN_SIZE - 1), (0, 255, 255), 1)
    cv2.line(zoom, (0, zoom_center), (ZOOM_WIN_SIZE - 1, zoom_center), (0, 255, 255), 1)

    cv2.imshow("Zoom", zoom)


def handle(event: int, x: int, y: int, flags: int, param: object) -> None:
    """Callback do mouse para captura de pontos na imagem.

    Args:
        event (int): Tipo do evento do mouse.
        x (int): Coordenada X do cursor na imagem exibida.
        y (int): Coordenada Y do cursor na imagem exibida.
        flags (int): Flags adicionais do evento do mouse (não utilizado).
        param (object): Parâmetro extra do callback do mouse.

    Returns:
        None: Atualiza a lista global de pontos clicados.

    """
    global clicked_points

    # Atualizar o zoom sempre que o do mouse se mover
    if event == cv2.EVENT_MOUSEMOVE:
        show_zoom(x, y)

    # Registra ponto clicado
    if event == cv2.EVENT_LBUTTONDOWN:
        x_original = int(x / SCALE)
        y_original = int(y / SCALE)

        clicked_points.append((x_original, y_original))
        print(f"Ponto selecionado: {(x_original, y_original)}")


def run_measurement() -> None:
    """Executa fluxo completo de medição.

    Etapas:
        1) Carrega modelo calibrado
        2) Seleciona imagem
        3) Coleta base e topo
        4) Calcula altura
        5) Exibe resultado

    """
    global original_img, clicked_points

    clicked_points = []

    # Carregar modelo
    model = load_model(MODEL_PATH)

    # Selecionar imagem
    image_path = select_image()

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise RuntimeError("Erro ao carregar imagem.")

    # Interface de seleção
    cv2.namedWindow("Measurement")
    cv2.setMouseCallback("Measurement", handle)

    print("\nClique em:")
    print("1) Base do objeto")
    print("2) Topo do objeto\n")

    # Lista para armazenar as medições realizadas
    measurements = []
    
    while True:
        temp = original_img.copy()

        # Desenha medições já realizadas
        for base, top, height in measurements:
            cv2.line(temp, base, top, (0, 0, 255), 3)
            mid = ((base[0] + top[0]) // 2, (base[1] + top[1]) // 2)
            cv2.putText(temp, f"{height:.2f} m", mid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Desenha pontos em seleção
        for point in clicked_points:
            cv2.circle(temp, point, 5, (255, 0, 255), -1)

        display_img = resize_for_display(temp, SCALE)
        cv2.imshow("Measurement", display_img)

        # Quando tiver um par de pontos
        if len(clicked_points) == 2:
            base, top = clicked_points

            # Estimar altura
            height = estimate_height(model, base, top)
            print(f"Altura = {height:.2f} m\n")

            # Armazenar medição
            measurements.append((base, top, height))

            # Limpa para próxima medição
            clicked_points = []

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            temp = original_img.copy()
            clicked_points = []
            measurements = []            
            print("Medições resetadas.\n")

    cv2.destroyAllWindows()

# ==============================
# EXECUÇÃO
# ==============================
if __name__ == "__main__":
    print("\n--- MEDIÇÃO DE ALTURA ---\n")
    run_measurement()