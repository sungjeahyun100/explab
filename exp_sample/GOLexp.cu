#include "database.hpp"
#include "perceptron.hpp"
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>

// 입력 패턴 크기 (10x10)
const int WIDTH = 10;
const int HEIGHT = 10;

// MLP 구조 정의
Adam inputlayer(WIDTH * HEIGHT, 256, 0.001, InitType::He);
ActivateLayer input(256, 1, ActivationType::LReLU);
Adam hiddenlayer1(256, 256, 0.001, InitType::He);
ActivateLayer hidden1(256, 1, ActivationType::LReLU);
Adam outputlayer(256, BOARDHEIGHT * BOARDWIDTH, 0.001, InitType::He);
ActivateLayer output(BOARDHEIGHT * BOARDWIDTH, 1, ActivationType::LReLU);
LossLayer loss(BOARDHEIGHT * BOARDWIDTH, 1, LossType::MSE);

// 데이터 로딩
std::vector<std::pair<d_matrix<double>, d_matrix<double>>> loadPatternData() {
    std::vector<std::pair<d_matrix<double>, d_matrix<double>>> dataset;
    for (int i = 1; i <= 40; ++i) {
        std::ifstream fin("dataset/sample" + std::to_string(i) + ".txt");
        if (!fin) continue;

        d_matrix<double> in(WIDTH * HEIGHT, 1);
        std::string line;
        for (int r = 0; r < HEIGHT; ++r) {
            std::getline(fin, line);
            for (int c = 0; c < WIDTH; ++c)
                in(r * WIDTH + c, 0) = line[c] - '0';
        }

        std::getline(fin, line); // label (사용하지 않음)
        std::getline(fin, line); // 빈 줄

        d_matrix<double> out(BOARDHEIGHT * BOARDWIDTH, 1);
        for (int r = 0; r < BOARDHEIGHT; ++r) {
            std::getline(fin, line);
            for (int c = 0; c < BOARDWIDTH; ++c)
                out(r * BOARDWIDTH + c, 0) = line[c] - '0';
        }

        dataset.emplace_back(in, out);
    }
    return dataset;
}

// 순전파 후 결과 반환
d_matrix<double> forward(const d_matrix<double>& in) {
    inputlayer.feedforward(in);
    input.pushInput(inputlayer.getOutput());
    input.Active();

    hiddenlayer1.feedforward(input.getOutput());
    hidden1.pushInput(hiddenlayer1.getOutput());
    hidden1.Active();

    outputlayer.feedforward(hidden1.getOutput());
    output.pushInput(outputlayer.getOutput());
    output.Active();

    return output.getOutput();
}

// 학습 루프
void train(std::vector<std::pair<d_matrix<double>, d_matrix<double>>>& data) {
    const int batch = 10;
    const int epoch = 1;

    for (int e = 0; e < epoch; ++e) {
        std::shuffle(data.begin(), data.end(), std::mt19937{e});
        for (size_t i = 0; i < data.size(); i += batch) {
            size_t end = std::min(i + batch, data.size());
            for (size_t j = i; j < end; ++j) {
                auto& in = data[j].first;
                auto& target = data[j].second;

                d_matrix<double> pred = forward(in);

                loss.pushTarget(target);
                loss.pushOutput(pred);
                d_matrix<double> grad = loss.getGrad();

                outputlayer.backprop(nullptr, grad, output.d_Active(outputlayer.getOutput()));

                d_matrix<double> dummy(hiddenlayer1.getOutput().getRow(), 1);
                dummy.fill(0);
                hiddenlayer1.backprop(&outputlayer, dummy, hidden1.d_Active(hiddenlayer1.getOutput()));
                inputlayer.backprop(&hiddenlayer1, dummy, input.d_Active(inputlayer.getOutput()));
            }
        }
    }
}

// 예측 결과 출력
void printBoard(const d_matrix<double>& board) {
    for (int r = 0; r < BOARDHEIGHT; ++r) {
        for (int c = 0; c < BOARDWIDTH; ++c) {
            std::cout << (board(r * BOARDWIDTH + c, 0) > 0.5 ? '1' : '0');
        }
        std::cout << '\n';
    }
}

int main() {
    auto dataset = loadPatternData();
    train(dataset);

    if (!dataset.empty()) {
        d_matrix<double> result = forward(dataset[0].first);
        result.cpyToHost();
        printBoard(result);
    }

    return 0;
}