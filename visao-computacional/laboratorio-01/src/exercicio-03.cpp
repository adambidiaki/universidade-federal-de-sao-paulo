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

    int x, y;

    cout << "Digite coordenada X: ";
    cin >> x;

    cout << "Digite coordenada Y: ";
    cin >> y;

    if (x >= 0 && x < grayImage.cols &&
        y >= 0 && y < grayImage.rows)
    {
        int pixelValue = grayImage.at<uchar>(y, x);
        cout << "Valor do pixel: " << pixelValue << endl;
    }
    else
    {
        cout << "Coordenada invalida!" << endl;
    }

    return 0;
}