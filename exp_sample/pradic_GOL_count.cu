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

    Adam layer1(100, 128, 0.0005, InitType::He);
    ActivateLayer act1(128, 1, ActivationType::LReLU);
    Adam layer2(128, 64, 0.0005, InitType::He);
    ActivateLayer act2(64, 1, ActivationType::LReLU);
    Adam outputLayer(64, BIT_WIDTH, 0.0005, InitType::He);
    ActivateLayer outAct(BIT_WIDTH, 1, ActivationType::LReLU);
    LossLayer loss(BIT_WIDTH, 1, LossType::MSE);

    const int epochs = 400;
    const int batchSize = 50;

    for(int epoch=0; epoch<epochs; ++epoch){
        auto startTime = std::chrono::steady_clock::now();

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

                auto Grad = loss.getGrad();
                auto Loss = loss.getLoss();

                outputLayer.backprop(nullptr, Grad, outAct.d_Active(outputLayer.getOutput()));
                d_matrix<double> dummy(1,1);
                layer2.backprop(&outputLayer, dummy, act2.d_Active(layer2.getOutput()));
                layer1.backprop(&layer2, dummy, act1.d_Active(layer1.getOutput()));

                printProgressBar(j, dataset.size(), startTime, "Epoch" + std::to_string(epoch+1) + " 진행중...(loss:" + std::to_string(Loss) + ")");
            }
        }
        std::cout << "✅ Epoch " << (epoch+1)
                  << " 완료! (소요 "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - startTime
                     ).count()
                  << "초),"
                  << "loss:"
                  << loss.getLoss()
                  << "                                                                                                                                          "
                  << std::endl;
    }

    for(size_t idx=0; idx<dataset.size(); ++idx){
        auto &inputMat = dataset[idx].first;
    
        // 순전파
        layer1.feedforward(inputMat);
        act1.pushInput(layer1.getOutput()); 
        act1.Active();
        layer2.feedforward(act1.getOutput());
        act2.pushInput(layer2.getOutput()); 
        act2.Active();
        outputLayer.feedforward(act2.getOutput());
        outAct.pushInput(outputLayer.getOutput()); 
        outAct.Active();
    
        // 예측값 복사
        d_matrix<double> pred = outAct.getOutput();
        pred.cpyToHost();
    
        // 비트 예측 및 정수값 복원
        int count = 0;
        for(int b = 0; b < BIT_WIDTH; ++b){
            // sigmoid 출력이니 0.5 기준으로 0/1 결정
            int bit = (pred(b,0) > 0.5) ? 1 : 0;
            count |= (bit << b);  // 2^b 만큼 더하기
        }
    
        // 결과 저장
        std::ofstream ofs(result_path + "/sample_count_bin_" + std::to_string(idx+1) + ".txt");
        if(!ofs){
            std::cerr << "파일 열기 실패: sample " << (idx+1) << "\n";
            continue;
        }
        ofs << "=== sample " << (idx+1) << " 결과 ===\n";
        ofs << count << "\n";
        ofs.close();
    }
    
    return 0;
}
