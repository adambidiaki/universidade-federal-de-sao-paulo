#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // Leitura da imagem
    cv::Mat image = cv::imread("../images/imagem.jpg");

    // Verificar se a imagem foi carregada
    if (image.empty())
    {
        std::cout << "Erro ao carregar a imagem!" << std::endl;
        return -1;
    }

    // Obter o shape (Altura, Largura, Canais)
    int altura = image.rows;
    int largura = image.cols;
    int canais = image.channels();

    // Apresenta no terminal a resolução e canais
    std::cout << "Formato da imagem: " << "(" << largura << ", " << altura << ", " << canais << ")" << std::endl;

    // Apresenta imagem
    cv::imshow("Imagem Original", image);

    // Esperar qualquer tecla para fechar a janela
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}