import cv2
import numpy as np
import math
import json
import random
from tkinter import Tk, filedialog

# ==============================
# CONFIGURAÇÕES
# ==============================
OUTPUT_PATH = "models/height_model.json"

# Parâmetros Canny / Hough
CANNY_THRESHOLD_1 = 350
CANNY_THRESHOLD_2 = 450
CANNY_EDGES = 5
CANNY_APERTURE_SIZE = 3
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 200

# Parâmetros RANSAC
RANSAC_ITERATIONS = 500
RANSAC_THRESHOLD = 5.0   # distância máxima (px) para considerar inlier
RANSAC_RATIO = 0.95      # fração mínima de inliers para parada antecipada

# Configurações do zoom
SCALE = 0.7
ZOOM_HALF_SIZE = 80  # raio do recorte na imagem original (px)
ZOOM_WIN_SIZE = 300  # tamanho da janela de zoom (px)

original_img = None
clicked_points = []


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


def detect_lines(img: np.ndarray) -> list:
    """Detecta linhas usando Canny e Transformada de Hough. As linhas detectadas 
    são filtradas para remover linhas aproximadamente verticais e horizontais.

    Args:
        img (np.ndarray): Imagem original.

    Returns:
        list[np.ndarray]: Lista de linhas no formato [[rho, theta]].

    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(img_gray, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2, CANNY_EDGES, CANNY_APERTURE_SIZE, L2gradient=True)

    # # Salva a imagem de bordas para inspeção
    # cv2.imwrite("debug_edges.png", img_edge)

    lines = cv2.HoughLines(img_edge, rho=HOUGH_RHO, theta=HOUGH_THETA, threshold=HOUGH_THRESHOLD)

    if lines is None:
        print("[AVISO] Nenhuma linha detectada pelo Hough. Tente reduzir HOUGH_THRESHOLD.")
        return []

    print(f"Linhas detectadas pelo Hough: {len(lines)}")

    valid_lines = []
    for line in lines:
        rho, theta = line[0]
        # Descartando linhas verticais e horizontais
        if (theta > 0.4 and theta < 1.47) or (theta > 1.67 and theta < 2.74):
            valid_lines.append(line)

    print(f"Após filtro angular: {len(valid_lines)} linhas")

    # Se o filtro for restritivo demais, usa todas as linhas e avisa
    if len(valid_lines) < 2:
        print("[AVISO] Filtro angular removeu linhas demais. Usando todas as linhas detectadas.")
        return list(lines)

    return valid_lines


def polar_to_cartesian(line: np.ndarray, length: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Converte linha polar (rho, theta) em dois pontos cartesianos para desenho.

    Args:
        line (np.ndarray): Linha no formato [[rho, theta]].
        length (int): Distância usada para gerar os pontos da reta.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: Dois pontos cartesianos usados para desenho.

    """
    rho, theta = line[0]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    pt1 = (int(x0 + length * (-b)), int(y0 + length * a))
    pt2 = (int(x0 - length * (-b)), int(y0 - length * a))

    return pt1, pt2


def find_intersection_point(line1: np.ndarray, line2: np.ndarray) -> tuple[int, int]:
    """Implementation is based on code from https://stackoverflow.com/questions/46565975,
    Original author: StackOverflow contributor alkasm 
    Find an intercept point of 2 lines model

    Args:
        line1 (np.ndarray): line using rho and theta (polar coordinates) to represent.
        line2 (np.ndarray): line using rho and theta (polar coordinates) to represent.

    Return:
        tuple[int, int]: x and y for the intersection point.

    """
    # rho and theta for each line
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    # Use formula from https://stackoverflow.com/a/383527/5087436 to solve for intersection between 2 lines 
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ]) 
    b = np.array([[rho1], [rho2]])
    det_A = np.linalg.det(A)

    if det_A != 0:
        x0, y0 = np.linalg.solve(A, b)
        # Round up x and y because pixel cannot have float number
        x0, y0 = int(np.round(np.asarray(x0).item())), int(np.round(np.asarray(y0).item()))
        return x0, y0
    else:
        return None


def find_dist_to_line(point: tuple[int, int], line: (np.ndarray)) -> float:
    """Implementation is based on Computer Vision material, owned by the University of Melbourne
    Find an intercept point of the line model with a normal from point to it, to calculate the
    distance between point and intercept
    
    Args:
        point (tuple[int, int]): the point using x and y to represent.
        line (np.ndarray): the line using rho and theta (polar coordinates) to represent.
    
    Return:
        float: the distance from the point to the line

    """
    x0, y0 = point
    rho, theta = line[0]
    m = (-1 * (np.cos(theta))) / np.sin(theta)
    c = rho / np.sin(theta)
    
    # intersection point with the model
    x = (x0 + m * y0 - m * c) / (1 + m**2)
    y = (m * x0 + (m**2) * y0 - (m**2) * c) / (1 + m**2) + c
    dist = math.sqrt((x - x0)**2 + (y - y0)**2)

    return dist


def RANSAC(
    lines: list[np.ndarray],
    ransac_iterations: int,
    ransac_threshold: float,
    ransac_ratio: float
) -> tuple[int, int]:
    """Implementation is based on code from Computer Vision material, owned by the University of Melbourne
    Use RANSAC to identify the vanishing points for a given image
    
    Args:
        lines (list[np.ndarray]): The lines for the image
        ransac_iterations (int): Maximum number of RANSAC iterations
        ransac_threshold (float): Maximum distance, in pixels, to consider a line as inlier
        ransac_ratio (float): RANSAC hyperparameters

    Return:
        tuple[int, int]: Estimated vanishing point for the image

    """
    inlier_count_ratio = 0.
    vanishing_point = (0, 0)

    # perform RANSAC iterations for each set of lines
    for iteration in range(ransac_iterations):
        # randomly sample 2 lines
        n = 2
        selected_lines = random.sample(lines, n)
        line1 = selected_lines[0]
        line2 = selected_lines[1]
        intersection_point = find_intersection_point(line1,line2)

        if intersection_point is not None:
            # count the number of inliers num
            inlier_count = 0
            # inliers are lines whose distance to the point is less than ransac_threshold
            for line in lines:
                # find the distance from the line to the point
                dist = find_dist_to_line(intersection_point,line)
                # check whether it's an inlier or not
                if dist < ransac_threshold:
                    inlier_count += 1

            # If the value of inlier_count is higher than previously saved value, save it, and save the current point
            if inlier_count/float(len(lines)) > inlier_count_ratio:
                inlier_count_ratio = inlier_count / float(len(lines))
                vanishing_point = intersection_point

            # We are done in case we have enough inliers
            if inlier_count > len(lines)*ransac_ratio:
                break

    return vanishing_point


def build_model(
    vp: tuple[float, float],
    base_ref: tuple[int, int],
    top_ref: tuple[int, int],
    real_height: float
) -> dict:
    """Cria estrutura do modelo de calibração.

    Args:
        vp (tuple[float, float]): Ponto de fuga.
        base_ref (tuple[int, int]): Base do objeto de referência.
        top_ref (tuple[int, int]): Topo do objeto de referência.
        real_height (float): Altura real do objeto (metros).

    Returns:
        dict: Estrutura serializável do modelo.

    """
    return {
        "vp": (float(vp[0]), float(vp[1])),
        "base_ref": (int(base_ref[0]), int(base_ref[1])),
        "top_ref": (int(top_ref[0]), int(top_ref[1])),
        "real_height": float(real_height)
    }


def save_model(model: dict, path: str) -> None:
    """Salva o modelo de calibração em um arquivo JSON.

    Args:
        model (dict): Estrutura do modelo.
        path (str): Caminho de saída.

    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)

    print(f"Modelo salvo em: {path}")


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


def run_calibration() -> None:
    """Executa o fluxo completo de calibração.

    Etapas:
        1) Detecta linhas com Canny + Hough Transform
        2) Filtra linhas verticais e horizontais
        3) Estima ponto de fuga com RANSAC
        4) Coleta objeto de referência
        5) Salva modelo calibrado

    """
    global original_img, clicked_points
    clicked_points = []

    # Selecionar imagem
    image_path = select_image()

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError("Imagem não encontrada.")

    # Criando uma cópia da imagem original
    draw_img = original_img.copy()

    # Detecção e filtragem de linhas
    lines = detect_lines(original_img)

    if not lines:
        raise RuntimeError("Nenhuma linha válida detectada.")

    # Desenha linhas filtradas
    for line in lines:
        pt1, pt2 = polar_to_cartesian(line, length=2000)
        cv2.line(draw_img, pt1, pt2, (0, 0, 255), 1)

    # # Salva draw_img antes do RANSAC para checar se linhas foram desenhadas
    # cv2.imwrite("debug_lines.png", draw_img)

    # Estimativa do ponto de fuga com RANSAC
    vp = RANSAC(lines, RANSAC_ITERATIONS, RANSAC_THRESHOLD, RANSAC_RATIO)

    if vp is None:
        raise RuntimeError("Não foi possível estimar o ponto de fuga.")

    print(f"Ponto de fuga estimado: {vp}")

    # Verificar se o ponto de fuga esta dentro dos limites da imagem
    height, width = draw_img.shape[:2]
    vp_in_frame = (0 <= vp[0] < width) and (0 <= vp[1] < height)
    if vp_in_frame:
        cv2.circle(draw_img, vp, 10, (0, 0, 255), -1)
        cv2.circle(draw_img, vp, 12, (255, 255, 255), 2)
    else:
        print(f"[AVISO] Vanish Point fora dos limites da imagem ({width}x{height}): {vp}. Não será desenhado.")

    # Interface de seleção da referência
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", handle)

    print("\nClique em:")
    print("  1) Base do objeto de referência")
    print("  2) Topo do objeto de referência")

    while True:
        temp = draw_img.copy()

        for point in clicked_points:
            cv2.circle(temp, point, 5, (255, 0, 255), -1)

        display_img = resize_for_display(temp, SCALE)
        cv2.imshow("Calibration", display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or len(clicked_points) == 2:
            break

    cv2.destroyAllWindows()

    if len(clicked_points) < 2:
        raise RuntimeError("Pontos insuficientes.")

    base_ref, top_ref = clicked_points

    # Coleta a altura real informada pelo usuário
    real_height = float(input("Altura real do objeto de referência (m): "))

    # Constroi e salva o modelo
    model = build_model(vp, base_ref, top_ref, real_height)
    save_model(model, OUTPUT_PATH)

# ==============================
# EXECUÇÃO
# ==============================
if __name__ == "__main__":
    print("\n--- CALIBRAÇÃO DE ALTURA ---\n")
    run_calibration()
