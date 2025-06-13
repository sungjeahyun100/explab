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
const std::string graph_path = "../graph/count_ver_loss";

int main(){
    auto dataset = LoadingData();

    std::filesystem::create_directories(result_path);
    std::filesystem::create_directories(graph_path); 
    const char *cmd = "find ../dataset/result -type f -delete";
    std::system(cmd);

    std::ofstream loss_ofs(graph_path + "/loss_data_He_LReLU_MSE_batch50.txt");  // ← 추가
    loss_ofs << "# epoch loss\n"; 

    Adam inputlayer(100, 512, 0.0001, InitType::He);
    ActivateLayer inputAct(512, 1, ActivationType::LReLU);
    Adam hiddenlayer1(512, 512, 0.0001, InitType::He);
    ActivateLayer hiddenAct1(512, 1, ActivationType::LReLU);
    Adam hiddenlayer2(512, 128, 0.0001, InitType::He);
    ActivateLayer hiddenAct2(128, 1, ActivationType::LReLU);
    Adam outputLayer(128, BIT_WIDTH, 0.0001, InitType::He);
    ActivateLayer outAct(BIT_WIDTH, 1, ActivationType::LReLU);
    LossLayer loss(BIT_WIDTH, 1, LossType::MSE);

    const int epochs = 100;
    const int batchSize = 50;
    std::mt19937 rng(std::random_device{}());
    
    for(int epoch = 0; epoch < epochs; ++epoch){
        auto startTime = std::chrono::steady_clock::now();
    
        // 1) 에폭 시작 시 한 번만 shuffle
        std::shuffle(dataset.begin(), dataset.end(), rng);
    
        double totalLoss = 0.0;
        size_t sampleCount = 0;
    
        // 2) 배치별 학습
        for(size_t i = 0; i < dataset.size(); i += batchSize){
            size_t end = std::min(i + batchSize, dataset.size());
    
            for(size_t j = i; j < end; ++j){
                auto &inputMat  = dataset[j].first;
                auto &targetMat = dataset[j].second;
    
                // (b) 순전파
                inputlayer.feedforward(inputMat);
                inputAct.pushInput(inputlayer.getOutput()); inputAct.Active();
                hiddenlayer1.feedforward(inputAct.getOutput());
                hiddenAct1.pushInput(hiddenlayer1.getOutput()); hiddenAct1.Active();
                hiddenlayer2.feedforward(hiddenAct1.getOutput());
                hiddenAct2.pushInput(hiddenlayer2.getOutput()); hiddenAct2.Active();
                outputLayer.feedforward(hiddenAct2.getOutput());
                outAct.pushInput(outputLayer.getOutput()); outAct.Active();
    
                // (c) 손실 계산
                loss.pushTarget(targetMat);
                loss.pushOutput(outAct.getOutput());
                double L = loss.getLoss();
                totalLoss += L;
                ++sampleCount;
    
                // (d) 역전파
                auto Grad = loss.getGrad();
                outputLayer.backprop(nullptr, Grad, outAct.d_Active(outputLayer.getOutput()));
                d_matrix<double> dummy(1,1);
                hiddenlayer2.backprop(&outputLayer, dummy, hiddenAct2.d_Active(hiddenlayer2.getOutput()));
                hiddenlayer1.backprop(&hiddenlayer2, dummy, hiddenAct1.d_Active(hiddenlayer1.getOutput()));
                inputlayer.backprop(&hiddenlayer1, dummy, inputAct.d_Active(inputlayer.getOutput()));
    
                // (e) 진행 표시
                printProgressBar(j, dataset.size(), startTime, "Epoch " + std::to_string(epoch+1) + " 진행중...(loss:" + std::to_string(totalLoss/1000) + ")");
            }
        }
    
        // 3) 에폭 단위 평균 손실 계산
        double avgLoss = totalLoss / static_cast<double>(sampleCount);

        loss_ofs << (epoch+1) << " " << avgLoss << "\n";
    
        std::cout << "✅ Epoch " << (epoch+1)
                  << " 완료! (소요 "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - startTime
                     ).count()
                  << "초), 평균 손실: "
                  << avgLoss
                  << "                                                                                                                                          "
                  << std::endl;
    }

    loss_ofs.close();


    for(size_t idx=0; idx<dataset.size(); ++idx){
        auto &inputMat = dataset[idx].first;
    
        // 순전파
        inputlayer.feedforward(inputMat);
        inputAct.pushInput(inputlayer.getOutput()); inputAct.Active();
        hiddenlayer1.feedforward(inputAct.getOutput());
        hiddenAct1.pushInput(hiddenlayer1.getOutput()); hiddenAct1.Active();
        hiddenlayer2.feedforward(hiddenAct1.getOutput());
        hiddenAct2.pushInput(hiddenlayer2.getOutput()); hiddenAct2.Active();
        outputLayer.feedforward(hiddenAct2.getOutput());
        outAct.pushInput(outputLayer.getOutput()); outAct.Active();
    
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
        ofs << pred(0, 0) << "," << pred(1, 0) << "," << pred(2, 0) << ","<< pred(3, 0) << "," << pred(4, 0) << "," << pred(5, 0) << "," << pred(6, 0) << "," << pred(7, 0) << "," << "\n";
        ofs.close();
    }
    
    return 0;
}
