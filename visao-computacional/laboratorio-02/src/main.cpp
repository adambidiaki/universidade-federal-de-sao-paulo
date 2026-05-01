#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;
namespace fs = filesystem;

int main()
{
    // Caminhos utilizados
    string imagePath  = "../images/";
    string outputPath = "../detections/";

    // Cria pasta de saída se não existir
    fs::create_directories(outputPath);

    // Imagens encontradas dentro do diretório
    vector<string> images;
    for (const auto& entry : fs::directory_iterator(imagePath))
    {
        if (entry.is_regular_file())
        {
            string ext = entry.path().extension().string();
            // Converte extensão para minúsculo antes de comparar
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg")
                images.push_back(entry.path().string());
        }
    }
    // Organizar em ordem alfabética
    sort(images.begin(), images.end());

    // Percorre todas as imagens
    for (const auto& image : images)
    {
        // Leitura da imagem
        Mat img = imread(image);

        if (img.empty())
        {
            cout << fs::path(image).filename().string() << ": erro ao carregar\n";
            continue;
        }

        // Pré-processamento utilizado para melhorar a detecção dos círculos
        Mat processedImage;
        bilateralFilter(img, processedImage, 9, 75, 75);
        // medianBlur(img, processedImage, 5);
        // GaussianBlur(img, processedImage, Size(9, 9), 2);

        Mat grayImage;
        cvtColor(processedImage, grayImage, COLOR_BGR2GRAY);

        // Detecta círculos usando o método de Hough
        vector<Vec3f> circles;
        HoughCircles(
            grayImage,
            circles,
            cv::HOUGH_GRADIENT,  // Método de detecção
            1.4,                 // Resolução de busca por cículos
            120,
            150,                 // Limite superior para o detector de bordas
            50,                  // Sensibilidade da detecção
            100,
            400
        );

        // Extrai o nome da imagem
        string image_name = fs::path(image).filename().string();

        if (!circles.empty())
        {
            // cout << image_name << ": tem circulo\n";

            // Desenhar os círculos
            for (const auto& circle_ : circles)
            {
                Point center(cvRound(circle_[0]), cvRound(circle_[1]));
                int radius = cvRound(circle_[2]);

                circle(img, center, radius, Scalar(0, 0, 255), 3);  // Desenhar o círculo encontrado
                // circle(img, center, 5, Scalar(0, 0, 255), -1);   // Desenhar centro do círculo
            }

            // Salvar a imagem com marcação
            imwrite(outputPath + image_name, img);
        }
        else
        {
            // Caso o círculo não seja encontrado
            cout << image_name << ": nao tem circulo\n";
            imwrite(outputPath + image_name, img);
        }
    }

    cout << "Deteccao de circulos finalizada\n";
    return 0;
}