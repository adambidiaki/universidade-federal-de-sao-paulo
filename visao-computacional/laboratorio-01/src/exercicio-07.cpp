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

    // Imagem reduzida pela metade
    Mat resizedImage;    
    resize(image, resizedImage, Size(), 0.5, 0.5, INTER_NEAREST);

    // Imagem reampliada para o tamanho original
    Mat expandedImage;    
    resize(resizedImage, expandedImage, image.size(), 0, 0, INTER_NEAREST);

    imshow("Original", image);
    imshow("Reduzida", resizedImage);
    imshow("Reampliada", expandedImage);

    waitKey(0);
    destroyAllWindows();

    return 0;
}