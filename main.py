import cv2
import subprocess
import os
import numpy as np

# Função para rodar o script aruco_detect.py

import cv2
import numpy as np
import os

#abrir a camera
#detectar o aruco -> salvar a imagem
#atraves da imagem -> obter as coordenadas do aruco 
#atraves da imagem -> fazer a mascara nos arucos (ou outro metodo que "retire" os arucos da imagme) -> salvar a imagem com a mascara 

def calcular_coordenadas_aruco(imagem_path):
    # Carregar a imagem salva
    img = cv2.imread(imagem_path)

    # Carregar parâmetros da câmera
    camera_matrix = np.load('tcc-residencia/intrinsic_extrinsic/camera_matrix.npy')
    dist_coeffs = np.load('tcc-residencia/intrinsic_extrinsic/dist_coefficients.npy')

    # Configurar o detector de ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()

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


def open_cam():
    # Carregar parâmetros da câmera
    camera_matrix = np.load('tcc-residencia/intrinsic_extrinsic/camera_matrix.npy')
    dist_coeffs = np.load('tcc-residencia/intrinsic_extrinsic/dist_coefficients.npy')
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    img_counter = 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Configurar o detector de ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    
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
                coordenadas_aruco = calcular_coordenadas_aruco(img_name)
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

 
def run_caminhos_de_corte():
    import cv2
    from ultralytics import YOLO
    from PIL import Image

    model = YOLO('best.pt')

    name = "output_images/opencv_frame_0000.png"


    mamepred = "b2pred.jpg"
    json_name = name + ".json"

    original_image = cv2.imread(name)

    for m in range(36, 37, 1):
        name = "output_images/opencv_frame_0000.png"
        mamepred = "b2pred.jpg"
        json_name = name + ".json"

        results = model(name)

        coord_x_pills = []
        coord_y_pills = []

        labels = []

        for r in results:
            labels = r.boxes.cls
            im_array = r.plot() 
            im = Image.fromarray(im_array[..., ::-1]) 
            im.save(mamepred)  
          
            cont = 0
            for box in r.boxes.xyxy:
                
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                # print(f"Bounding Box Center + {cont}:({x_center}, {y_center})")
                print(labels[cont])
                if labels[cont] == 0:
                    cord = ()
                    for i in box:
                        cord = cord + (int(i),)

                    print(cord)
                    roi = original_image[cord[1]: cord[3], cord[0]: cord[2]]
                    roi_filename = f"roi_pill_{cont}.jpg"
                    cv2.imwrite(roi_filename, roi)
                    cv2.imshow(f"ROI {cont}", roi)
                    cv2.waitKey(0)  
                    cv2.destroyAllWindows()

                if labels[cont] == 1:
                    coord_x_pills.append(x_center)
                    coord_y_pills.append(y_center)

                cont = cont + 1

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.spatial import Voronoi, voronoi_plot_2d

        coordinates_pred = np.empty(0)
        list_coord = []

        for ni in range(0, len(coord_x_pills)):
            list_coord.append([coord_x_pills[ni], coord_y_pills[ni]])

        # print("coordinates:",  list_coord)

        # Defina os pontos geradores de Voronoi (você precisa especificar esses pontos)
        points = np.array(list_coord, dtype=np.float64)

        vor = Voronoi(points)
        # print(vor.ridge_vertices)q

        # Plote o diagrama de Voronoi
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=0.4, point_size=1)

        # Carregue sua imagem de fundo
        background_image = plt.imread(name)  # Substitua pelo caminho da sua imagem

        # Inverta o eixo Y para que a origem esteja no canto superior esquerdo
        # ax.invert_yaxis()

        # Adicione a imagem de fundo ao gráfico
        ax.imshow(background_image, origin='upper')
        nome_da_imagem = "Voronoi f (" + str(m) + ").png"

        plt.ylim([0, 3000])  # Ajuste os limites do gráfico conforme necessário
        plt.xlim([0, 4000])

        # plt.show()
        fig.savefig(nome_da_imagem, dpi=500)

def main():   
    #run_caminhos_de_corte()
    open_cam()
    imagem_path = "output_images/opencv_frame_0000_with_aruco.png"  # Caminho para a imagem salva
    coordenadas_aruco = calcular_coordenadas_aruco(imagem_path)

    if coordenadas_aruco is not None:
        for i, coord in enumerate(coordenadas_aruco):
            print(f"Coordenadas do ArUco {i + 1}: {coord}")
    
    if frame is not None:
        #run_aruco_detect(frame)  # Passa a imagem para detecção ArUco
        run_caminhos_de_corte()
    else:
        print("Nenhuma imagem foi capturada.")


def get_coordinates(coor_phill, coor_cut, coor_aruco):
    pass


if __name__ == "__main__":
    main()
