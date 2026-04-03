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

    // Imagem reduzida pela metade
    Mat resizedImage;    
    resize(grayImage, resizedImage, Size(), 0.5, 0.5, INTER_NEAREST);

    // Imagem com poucos níveis de cinza
    Mat quantizedImage = grayImage.clone();
    int grayLevel = 256 / 4;

    for (int y = 0; y < grayImage.rows; y++)
    {
        for (int x = 0; x < grayImage.cols; x++)
        {
            int pixel = grayImage.at<uchar>(y, x);
            pixel = (pixel / grayLevel) * grayLevel;

            quantizedImage.at<uchar>(y, x) = pixel;
        }
    }

    // Imagem reduzida e com poucos níveis de cinza
    Mat transformedImage;    
    resize(quantizedImage, transformedImage, Size(), 0.5, 0.5, INTER_NEAREST);

    imshow("Original", grayImage);
    imshow("Reduzida", resizedImage);
    imshow("Quantizada", quantizedImage);
    imshow("Reduzida e Quantizada", transformedImage);

    waitKey(0);
    destroyAllWindows();

    return 0;
}

/*
A imagem reduzida apresenta perda de detalhes devido à diminuição da resolução. Já a imagem 
com poucos tons de cinza apresenta perda de detalhes nos tons de cinza. A imagem reduzida e 
com poucos níveis de cinza apresenta ambos os efeitos, resultando em uma imagem de menor
qualidade quando comparada com a imagem original.
*/