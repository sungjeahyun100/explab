#include <database.hpp>
#include <perceptron.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

const std::string result_path = "../dataset/result";

int main(){
    auto dataset = LoadingData();

    std::filesystem::create_directories(result_path);
    const char *cmd = "find ../dataset/result -type f -delete";
    std::system(cmd);

    Adam layer1(100, 128, 0.01, InitType::Xavier);
    ActivateLayer act1(128, 1, ActivationType::Tanh);
    Adam layer2(128, 64, 0.01, InitType::Xavier);
    ActivateLayer act2(64, 1, ActivationType::Tanh);
    Adam outputLayer(64, 1, 0.01, InitType::Xavier);
    ActivateLayer outAct(1, 1, ActivationType::Identity);
    LossLayer loss(1, 1, LossType::MSE);

    const int epochs = 20;
    const int batchSize = 10;
    std::mt19937 rng(std::random_device{}());

    for(int epoch=0; epoch<epochs; ++epoch){
        auto startTime = std::chrono::steady_clock::now();
        std::shuffle(dataset.begin(), dataset.end(), rng);

        for(size_t i=0; i<dataset.size(); i+=batchSize){
            size_t end = std::min(i+batchSize, dataset.size());
            for(size_t j=i; j<end; ++j){
                auto &inputMat = dataset[j].first;
                auto &targetMat = dataset[j].second;

                layer1.feedforward(inputMat);
                act1.pushInput(layer1.getOutput());
                act1.Active();

                layer2.feedforward(act1.getOutput());
                act2.pushInput(layer2.getOutput());
                act2.Active();

                outputLayer.feedforward(act2.getOutput());
                outAct.pushInput(outputLayer.getOutput());
                outAct.Active();

                loss.pushTarget(targetMat);
                loss.pushOutput(outAct.getOutput());

                outputLayer.backprop(nullptr, loss.getGrad(), outAct.d_Active(outputLayer.getOutput()));
                d_matrix<double> dummy(1,1);
                layer2.backprop(&outputLayer, dummy, act2.d_Active(layer2.getOutput()));
                layer1.backprop(&layer2, dummy, act1.d_Active(layer1.getOutput()));

                printProgressBar(j, dataset.size(), startTime, "Epoch" + std::to_string(epoch+1) + " 진행중...(loss:" + std::to_string(loss.getLoss()) + ")");
            }
        }
        std::cout << "\n✅ Epoch " << (epoch+1) << " 완료" << std::endl;
    }

    for(size_t idx=0; idx<dataset.size(); ++idx){
        auto &inputMat = dataset[idx].first;

        layer1.feedforward(inputMat);
        act1.pushInput(layer1.getOutput());
        act1.Active();
        layer2.feedforward(act1.getOutput());
        act2.pushInput(layer2.getOutput());
        act2.Active();
        outputLayer.feedforward(act2.getOutput());
        outAct.pushInput(outputLayer.getOutput());
        outAct.Active();

        d_matrix<double> pred = outAct.getOutput();
        pred.cpyToHost();
        int count = static_cast<int>(std::round(pred(0,0)));

        std::ofstream ofs(result_path + "/sample_count_ver_" + std::to_string(idx+1) + ".txt");
        ofs << "=== sample " << idx+1 << " 결과 ===\n";
        ofs << count << "\n";
        ofs.close();
    }

    return 0;
}
