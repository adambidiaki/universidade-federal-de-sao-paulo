#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // Leitura da imagem
    Mat image = imread("../images/imagem.jpg");

    if (image.empty())
    {
        cout << "Erro ao carregar a imagem!" << endl;
        return -1;
    }

    // Converter para escala de cinza
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Apresenta imagens
    imshow("Imagem Original", image);
    imshow("Imagem em Cinza", grayImage);

    // Esperar qualquer tecla para fechar a janela
    waitKey(0);
    destroyAllWindows();

    return 0;
}