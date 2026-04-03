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

    // Binarizar imagem
    Mat binaryImage;
    threshold(grayImage, binaryImage, 128, 255, THRESH_BINARY);

    imshow("Original", grayImage);
    imshow("Binaria", binaryImage);

    waitKey(0);
    destroyAllWindows();

    return 0;
}