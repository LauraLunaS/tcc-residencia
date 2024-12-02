import cv2
from ultralytics import YOLO
from PIL import Image

model = YOLO('best.pt')

name = "output_images/opencv_frame_0000.png"

mamepred = "b2pred.jpg"
json_name = name + ".json"

original_image = cv2.imread(name)


def run_caminhos_de_corte():
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
