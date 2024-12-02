import os
from parameters import *
from coordinates_aruco import get_cordinates


def open_cam():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    img_counter = 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Criar pasta de saída para salvar as imagens processadas
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        # Capturar o quadro da câmera
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o quadro.")
            break

        # Corrigir a distorção da imagem
        h, w = frame.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Detectar os marcadores ArUco na imagem corrigida
        corners, ids, rejected = cv2.aruco.detectMarkers(undistorted_img, aruco_dict, parameters=aruco_params)

        # Criar uma cópia da imagem para desenhar os ArUcos detectados
        aruco_detected_img = undistorted_img.copy()

        # Criar uma máscara inicial (toda branca)
        mask = np.ones(undistorted_img.shape[:2], dtype=np.uint8) * 255  # 255 é branco

        if ids is not None:
            for i, corner in enumerate(corners):
                # Os pontos de cada marcador estão em 'corner[0]', que é um array de 4 pontos
                pts = np.int32(corner[0])  # Converte para inteiros
                # Nesses pontos, temos as coordenadas dos 4 cantos do ArUco, ou seja, temos o tamnho do ArUco para
                # comparar com o tamanho do blíster

                # Preencher a área do marcador com 0 (preto) na máscara
                cv2.fillPoly(mask, [pts], 0)

                # Desenhar os marcadores detectados na imagem
                cv2.aruco.drawDetectedMarkers(aruco_detected_img, corners, ids)

            # Obter as coordenadas dos ArUcos
            for i, corner in enumerate(corners):
                print(f"Coordenadas do ArUco {ids[i]}: {corner[0]}")

            # Aplicar a máscara para remover os ArUcos da imagem
            masked_img = cv2.bitwise_and(undistorted_img, undistorted_img, mask=mask)

            # Exibir as janelas
            cv2.imshow("ArUcos Detectados", aruco_detected_img)
            cv2.imshow("Imagem com Mascara", masked_img)

        else:
            # Exibir a imagem sem marcadores detectados
            cv2.imshow("ArUcos Detectados", undistorted_img)
            cv2.imshow("Imagem com Mascara", undistorted_img)

        # Aguarda a tecla pressionada
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressionado para sair
            print("Saindo...")
            break
        elif k % 256 == 32:  # SPACE pressionado para salvar
            if ids is not None:
                # Salvar a imagem com os ArUcos detectados
                img_name = os.path.join(output_dir, f"opencv_frame_{img_counter:04d}_with_aruco.png")
                cv2.imwrite(img_name, aruco_detected_img)
                print(f"Imagem com ArUcos detectados salva: {img_name}")

                # Salvar a imagem com a máscara aplicada (sem os ArUcos)
                masked_img_name = os.path.join(output_dir, f"opencv_frame_{img_counter:04d}_masked.png")
                cv2.imwrite(masked_img_name, masked_img)
                print(f"Imagem com máscara salva: {masked_img_name}")

                # Calcular as coordenadas dos ArUcos na imagem salva
                coordenadas_aruco = get_cordinates(img_name)
                if coordenadas_aruco:
                    print("Coordenadas dos ArUcos na imagem salva:")
                    for coords in coordenadas_aruco:
                        print(coords)

                img_counter += 1
            else:
                print("Nenhum ArUco detectado. Nenhuma imagem salva.")

    # Liberar os recursos após o loop
    cap.release()
    cv2.destroyAllWindows()
