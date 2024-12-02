from parameters import *


def get_cordinates(imagem_path):
    # Carregar a imagem salva
    img = cv2.imread(imagem_path)

    # Corrigir distorção (se necessário)
    h, w = img.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Detectar os marcadores ArUco
    corners, ids, rejected = cv2.aruco.detectMarkers(undistorted_img, aruco_dict, parameters=aruco_params)

    # Verificar se ArUcos foram detectados
    if ids is not None:
        coordenadas = []
        # Para cada ArUco detectado, calcular as coordenadas
        for i, corner in enumerate(corners):
            print(f"ArUco {ids[i]} detectado.")
            print(f"Coordenadas do ArUco {ids[i]}: {corner[0]}")  # As coordenadas dos 4 cantos do ArUco
            coordenadas.append(corner[0])  # Adiciona as coordenadas dos 4 cantos

        return coordenadas
    else:
        print("Nenhum ArUco detectado na imagem.")
        return None
