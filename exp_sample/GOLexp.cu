#include "database.hpp"
#include "perceptron.hpp"

Adam inputlayer(100, 256, 0.001, InitType::He);
ActivateLayer input(256, 1, ActivationType::LReLU);
Adam hiddenlayer1(256, 256, 0.001, InitType::He);
ActivateLayer hidden1(256, 1, ActivationType::LReLU);
Adam outputlayer(256, BOARDHEIGHT*BOARDWIDTH, 0.001, InitType::He);
ActivateLayer output(BOARDHEIGHT*BOARDWIDTH, 1, ActivationType::LReLU);
LossLayer loss(BOARDHEIGHT*BOARDWIDTH, 1, LossType::MSE);



// 데이터셋 로드 함수: dataset/sampleN.txt 파일에서
// 10x10 입력 패턴과 300x300 결과 패턴을 읽어온다.
std::vector<std::pair<d_matrix<double>, d_matrix<double>>> loadPatternData(int count){
    std::vector<std::pair<d_matrix<double>, d_matrix<double>>> data;
    data.reserve(count);

    for(int idx = 1; idx <= count; ++idx){
        std::string path = "dataset/sample" + std::to_string(idx) + ".txt";
        std::ifstream fin(path);
        if(!fin){
            std::cerr << "파일을 열 수 없습니다: " << path << std::endl;
            continue;
        }

        d_matrix<double> inputMat(100,1);      // 10x10 패턴
        d_matrix<double> outputMat(BOARDHEIGHT*BOARDWIDTH,1); // 300x300 결과 패턴

        std::string line;
        // 입력 패턴 읽기
        for(int r=0;r<10 && std::getline(fin,line); ++r){
            for(int c=0;c<10 && c<(int)line.size(); ++c){
                inputMat(r*10+c,0) = line[c]-'0';
            }
        }

        // 라벨(생존 칸 수)은 현재 사용하지 않으므로 읽기만 하고 무시
        std::getline(fin,line); // label
        std::getline(fin,line); // blank line

        // 결과 패턴 읽기 (300줄)
        for(int r=0;r<BOARDHEIGHT && std::getline(fin,line); ++r){
            for(int c=0;c<BOARDWIDTH && c<(int)line.size(); ++c){
                outputMat(r*BOARDWIDTH+c,0) = line[c]-'0';
            }
        }

        data.emplace_back(inputMat, outputMat);
    }

    return data;
}

int main(){
    // 데이터셋 로드 (sample1 ~ sample40)
    auto dataset = loadPatternData(40);

    const int epochs = 5;           // 에폭 수 (예시)
    const int batchSize = 10;       // 미니배치 크기

    std::mt19937 rng(std::random_device{}());

    // 학습 루프
    for(int epoch=0; epoch<epochs; ++epoch){
        std::shuffle(dataset.begin(), dataset.end(), rng);

        for(size_t i=0; i<dataset.size(); i += batchSize){
            size_t end = std::min(i+batchSize, dataset.size());
            for(size_t j=i; j<end; ++j){
                auto& inputMat = dataset[j].first;
                auto& targetMat = dataset[j].second;

                // 순전파
                inputlayer.feedforward(inputMat);
                input.pushInput(inputlayer.getOutput());
                input.Active();

                hiddenlayer1.feedforward(input.getOutput());
                hidden1.pushInput(hiddenlayer1.getOutput());
                hidden1.Active();

                outputlayer.feedforward(hidden1.getOutput());
                output.pushInput(outputlayer.getOutput());
                output.Active();

                // 손실 계산 및 역전파
                loss.pushTarget(targetMat);
                loss.pushOutput(output.getOutput());
                d_matrix<double> grad = loss.getGrad();

                outputlayer.backprop(nullptr, grad, output.d_Active(outputlayer.getOutput()));
                // 다음 계층 정보를 포인터로 전달하면 내부에서 next->delta를 활용하여
                // 역전파가 진행된다. 외부에서 delta에 접근할 필요는 없다.
                d_matrix<double> dummy(1,1); // 사용되지 않음
                hiddenlayer1.backprop(&outputlayer, dummy, hidden1.d_Active(hiddenlayer1.getOutput()));
                inputlayer.backprop(&hiddenlayer1, dummy, input.d_Active(inputlayer.getOutput()));
            }
        }

        std::cout << "[Epoch " << epoch+1 << "] 학습 완료" << std::endl;
    }

    // 학습 후 각 샘플에 대한 예측 결과 출력
    for(size_t idx=0; idx<dataset.size(); ++idx){
        auto& inputMat = dataset[idx].first;

        inputlayer.feedforward(inputMat);
        input.pushInput(inputlayer.getOutput());
        input.Active();
        hiddenlayer1.feedforward(input.getOutput());
        hidden1.pushInput(hiddenlayer1.getOutput());
        hidden1.Active();
        outputlayer.feedforward(hidden1.getOutput());
        output.pushInput(outputlayer.getOutput());
        output.Active();

        d_matrix<double> pred = output.getOutput();
        pred.cpyToHost();

        std::cout << "=== sample " << idx+1 << " 결과 ===" << std::endl;
        for(int r=0;r<BOARDHEIGHT;r++){
            for(int c=0;c<BOARDWIDTH;c++){
                std::cout << (pred(r*BOARDWIDTH+c,0) > 0.5 ? '1' : '0');
            }
            std::cout << '\n';
        }
        std::cout << std::endl;
    }

    return 0;
}