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
        cout << "Erro ao carregar imagem!" << endl;
        return -1;
    }

    // Converter para escala de cinza
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Clonar a imagem original
    Mat negativeImage = grayImage.clone();

    // Criar negativo
    for (int y = 0; y < grayImage.rows; y++)
    {
        for (int x = 0; x < grayImage.cols; x++)
        {
            negativeImage.at<uchar>(y, x) = 255 - grayImage.at<uchar>(y, x);
        }
    }

    imshow("Original", grayImage);
    imshow("Negativo", negativeImage);

    waitKey(0);
    destroyAllWindows();

    return 0;
}