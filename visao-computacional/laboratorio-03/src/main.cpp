#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Configurações
const string BASE_DIR = "../images";
const string OUTPUT_DIR = "../results";
const vector<int> SCALES = [] {
    vector<int> s;
    for (int i = 1; i <= 20; i++) s.push_back(i);
    return s;
}();
const int SHAPE = MORPH_RECT;

// Carregamento de imagens
map<string, vector<Mat>> loadImages(const string& baseDir)
{
    map<string, vector<Mat>> classes;

    for (const auto& classEntry : fs::directory_iterator(baseDir))
    {
        if (!classEntry.is_directory())
            continue;

        string name = classEntry.path().filename().string();
        vector<Mat> images;

        vector<string> extensions = { ".png", ".jpg", ".jpeg", ".bmp", ".pgm" };

        // Coleta e ordena todos os arquivos válidos da pasta
        vector<fs::path> files;
        for (const auto& file : fs::directory_iterator(classEntry.path()))
        {
            string ext = file.path().extension().string();
            if (find(extensions.begin(), extensions.end(), ext) != extensions.end())
                files.push_back(file.path());
        }
        sort(files.begin(), files.end());

        for (const auto& filePath : files)
        {
            Mat img = imread(filePath.string(), IMREAD_GRAYSCALE);
            if (img.empty())
                continue;

            // Converte para float64 para cálculos de energia
            Mat imgFloat;
            img.convertTo(imgFloat, CV_64F);
            images.push_back(imgFloat);
        }

        if (!images.empty())
        {
            classes[name] = images;
            cout << "  Classe '" << name << "': " << images.size() << " imagem(ns) carregada(s)\n";
        }
    }

    return classes;
}

// Cálculo das assinaturas
map<string, vector<vector<double>>> computeSignatures(
    const map<string, vector<Mat>>& classes,
    const vector<int>& scales,
    int shape)
{
    map<string, vector<vector<double>>> results;

    for (const auto& [className, images] : classes)
    {
        cout << "Processando classe '" << className << "'...\n";

        vector<vector<double>> classSignatures;

        for (const Mat& img : images)
        {
            double originalEnergy = sum(img)[0];
            double prevEnergy = originalEnergy;

            vector<double> signature;

            for (int scale : scales)
            {
                Mat kernel = getStructuringElement(shape, Size(scale, scale));

                // Abertura morfológica
                Mat opened;
                morphologyEx(img, opened, MORPH_OPEN, kernel);
                opened.convertTo(opened, CV_64F);

                double currentEnergy = sum(opened)[0];

                // Diferença de energia removida
                double diff = prevEnergy - currentEnergy;
                signature.push_back(diff);
                prevEnergy = currentEnergy;
            }

            // Normalização pela energia original
            if (originalEnergy > 0)
                for (double& v : signature)
                    v /= originalEnergy;

            classSignatures.push_back(signature);
        }

        results[className] = classSignatures;
    }

    return results;
}

// Exporta os resultados para .csv
void exportCSV(
    const map<string, vector<vector<double>>>& results,
    const vector<int>& scales,
    const string& outputDir)
{
    string path = outputDir + "/assinaturas.csv";
    ofstream file(path);

    // Cabeçalho: identificadores + uma coluna por escala
    file << "class,image_idx";
    for (int s : scales) file << ",scale_" << s;
    file << "\n";

    // Uma linha por imagem
    for (const auto& [name, signatures] : results)
    {
        for (int idx = 0; idx < (int)signatures.size(); idx++)
        {
            file << name << "," << idx;
            for (double v : signatures[idx])
                file << "," << v;
            file << "\n";
        }
    }

    cout << "  Salvo: " << path << "\n";
}

// Converte valor do eixo Y para coordenada de pixel no canvas
int toPixelY(double val, double minVal, double maxVal, int top, int height)
{
    return top + height - (int)((val - minVal) / (maxVal - minVal) * height);
}

// Converte índice X para coordenada de pixel no canvas
int toPixelX(int idx, int n, int left, int width)
{
    return left + (int)((double)idx / (n - 1) * width);
}

// Salvar as assinaturas como imagem PNG usando OpenCV
void plotSignatures(
    const map<string, vector<vector<double>>>& results,
    const vector<int>& scales,
    const string& outputDir)
{
    // Paleta de cores (BGR para OpenCV)
    vector<Scalar> colors = {
        Scalar(70,  57, 230),   // #e63946
        Scalar(143, 157, 42),   // #2a9d8f
        Scalar(97, 162, 244),   // #f4a261
        Scalar(155, 123, 69),   // #457b9d
        Scalar(65,  76, 109),   // #6d4c41
        Scalar(170, 36, 142),   // #8e24aa
    };

    int nClasses = (int)results.size();
    int W = 600, H = 350;
    int PAD_L = 55, PAD_R = 20, PAD_T = 40, PAD_B = 50;
    int cW = W - PAD_L - PAD_R;
    int cH = H - PAD_T - PAD_B;
    int N = (int)scales.size();

    // Encontra o valor máximo global para escalar o eixo Y
    double globalMax = 0;
    for (const auto& [name, sigs] : results)
        for (const auto& sig : sigs)
            for (double v : sig)
                globalMax = max(globalMax, v);
    globalMax *= 1.1;

    // Gráfico 1: assinaturas individuais (um subplot por classe)
    int totalW = W * nClasses;
    Mat canvas1(H, totalW, CV_8UC3, Scalar(255, 255, 255));

    int classIdx = 0;
    for (const auto& [name, signatures] : results)
    {
        int offsetX = classIdx * W;
        Scalar color = colors[classIdx % colors.size()];

        // Grade de fundo
        for (int g = 0; g <= 4; g++)
        {
            int y = PAD_T + (int)((double)g / 4 * cH);
            line(canvas1, Point(offsetX + PAD_L, y), Point(offsetX + PAD_L + cW, y),
                 Scalar(220, 220, 220), 1);
        }

        // Eixos
        line(canvas1, Point(offsetX+PAD_L, PAD_T), Point(offsetX+PAD_L, PAD_T+cH),
             Scalar(180,180,180), 1);
        line(canvas1, Point(offsetX+PAD_L, PAD_T+cH), Point(offsetX+PAD_L+cW, PAD_T+cH),
             Scalar(180,180,180), 1);

        // Labels eixo X
        for (int r : {1, 5, 10, 15, 20})
        {
            int x = offsetX + toPixelX(r-1, N, PAD_L, cW);
            putText(canvas1, to_string(r), Point(x-6, PAD_T+cH+16),
                    FONT_HERSHEY_SIMPLEX, 0.32, Scalar(120,120,120), 1);
        }
        putText(canvas1, "Raio", Point(offsetX+PAD_L+cW/2-12, H-8),
                FONT_HERSHEY_SIMPLEX, 0.35, Scalar(100,100,100), 1);

        // Título da classe
        putText(canvas1, "Classe: " + name, Point(offsetX+PAD_L+4, PAD_T-10),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(30,30,30), 1);

        // Assinaturas individuais
        for (const auto& sig : signatures)
        {
            for (int i = 1; i < N; i++)
            {
                Point p1(offsetX + toPixelX(i-1, N, PAD_L, cW),
                         toPixelY(sig[i-1], 0, globalMax, PAD_T, cH));
                Point p2(offsetX + toPixelX(i,   N, PAD_L, cW),
                         toPixelY(sig[i],   0, globalMax, PAD_T, cH));
                line(canvas1, p1, p2, color, 1, LINE_AA);
            }
        }

        // Média da classe
        vector<double> mean(N, 0);
        for (const auto& sig : signatures)
            for (int i = 0; i < N; i++) mean[i] += sig[i];
        for (double& v : mean) v /= signatures.size();

        for (int i = 1; i < N; i++)
        {
            Point p1(offsetX + toPixelX(i-1, N, PAD_L, cW),
                     toPixelY(mean[i-1], 0, globalMax, PAD_T, cH));
            Point p2(offsetX + toPixelX(i,   N, PAD_L, cW),
                     toPixelY(mean[i],   0, globalMax, PAD_T, cH));
            line(canvas1, p1, p2, Scalar(30,30,30), 2, LINE_AA);
        }

        classIdx++;
    }

    string path1 = outputDir + "/assinaturas_individuais.png";
    imwrite(path1, canvas1);
    cout << "  Salvo: " << path1 << "\n";

    // Gráfico 2: comparação das médias
    Mat canvas2(H, W, CV_8UC3, Scalar(255, 255, 255));

    for (int g = 0; g <= 4; g++)
    {
        int y = PAD_T + (int)((double)g / 4 * cH);
        line(canvas2, Point(PAD_L, y), Point(PAD_L+cW, y), Scalar(220,220,220), 1);
    }
    line(canvas2, Point(PAD_L, PAD_T), Point(PAD_L, PAD_T+cH), Scalar(180,180,180), 1);
    line(canvas2, Point(PAD_L, PAD_T+cH), Point(PAD_L+cW, PAD_T+cH), Scalar(180,180,180), 1);

    for (int r : {1, 5, 10, 15, 20})
    {
        int x = toPixelX(r-1, N, PAD_L, cW);
        putText(canvas2, to_string(r), Point(x-6, PAD_T+cH+16),
                FONT_HERSHEY_SIMPLEX, 0.32, Scalar(120,120,120), 1);
    }
    putText(canvas2, "Raio", Point(PAD_L+cW/2-12, H-8),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(100,100,100), 1);
    putText(canvas2, "Comparacao das Assinaturas Granulometricas",
            Point(PAD_L, PAD_T-10), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(30,30,30), 1);

    classIdx = 0;
    int legendY = PAD_T + 10;
    for (const auto& [name, signatures] : results)
    {
        Scalar color = colors[classIdx % colors.size()];

        vector<double> mean(N, 0);
        for (const auto& sig : signatures)
            for (int i = 0; i < N; i++) mean[i] += sig[i];
        for (double& v : mean) v /= signatures.size();

        for (int i = 1; i < N; i++)
        {
            Point p1(toPixelX(i-1, N, PAD_L, cW), toPixelY(mean[i-1], 0, globalMax, PAD_T, cH));
            Point p2(toPixelX(i,   N, PAD_L, cW), toPixelY(mean[i],   0, globalMax, PAD_T, cH));
            line(canvas2, p1, p2, color, 2, LINE_AA);
        }

        // Legenda
        rectangle(canvas2, Point(PAD_L+cW-120, legendY-8),
                  Point(PAD_L+cW-106, legendY+4), color, FILLED);
        putText(canvas2, name, Point(PAD_L+cW-102, legendY+4),
                FONT_HERSHEY_SIMPLEX, 0.32, Scalar(40,40,40), 1);
        legendY += 18;

        classIdx++;
    }

    string path2 = outputDir + "/assinaturas_comparacao.png";
    imwrite(path2, canvas2);
    cout << "  Salvo: " << path2 << "\n";
}

// Execução principal
int main()
{
    cout << "\n--- LABORATORIO 03 ---\n\n";

    fs::create_directories(OUTPUT_DIR);

    cout << "1. Carregando imagens...\n";
    auto classes = loadImages(BASE_DIR);

    cout << "\n2. Calculando assinaturas granulometricas...\n";
    auto results = computeSignatures(classes, SCALES, SHAPE);

    cout << "\n3. Gerando graficos...\n";
    plotSignatures(results, SCALES, OUTPUT_DIR);

    cout << "\n4. Exportando CSV...\n";
    exportCSV(results, SCALES, OUTPUT_DIR);

    cout << "\nConcluido! Resultados em '" << OUTPUT_DIR << "/'\n\n";
    return 0;
}