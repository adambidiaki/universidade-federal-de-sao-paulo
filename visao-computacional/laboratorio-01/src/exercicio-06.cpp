#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // Leitura da imagem já em escala de cinza
    Mat grayImage = imread("../images/imagem.jpg", IMREAD_GRAYSCALE);

    if (grayImage.empty())
    {
        cout << "Erro ao carregar imagem!" << endl;
        return -1;
    }

    // Apresentar imagem original
    imshow("Original", grayImage);

    // Níveis de cinza
    int levelsList[] = {128, 64, 16, 4};

    // Loop para cada nível
    for (int i = 0; i < 4; i++)
    {
        int levels = levelsList[i];

        Mat quantizedImage = grayImage.clone();
        int grayLevel = 256 / levels;

        for (int y = 0; y < grayImage.rows; y++)
        {
            for (int x = 0; x < grayImage.cols; x++)
            {
                int pixel = grayImage.at<uchar>(y, x);
                pixel = (pixel / grayLevel) * grayLevel;

                quantizedImage.at<uchar>(y, x) = pixel;
            }
        }

        // Alterando o nome da janela e exibindo a imagem
        string windowName = "Niveis de cinza: " + to_string(levels);
        imshow(windowName, quantizedImage);
    }

    waitKey(0);
    destroyAllWindows();

    return 0;
}

/*
Conforme o número de níveis de cinza é reduzido, a imagem perde detalhes nos tons de cinza.
Com níveis mais altos, como 128 e 64, a imagem mantém uma boa qualidade. Já com níveis mais baixos,
como 16 e 4, começam a aparecer faixas entre os tons e a imagem perde muitos detalhes.
*/