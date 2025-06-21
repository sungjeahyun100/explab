#include <d_matrix.hpp>
#include <perceptron.hpp>
#include <unordered_map>

std::vector<d_matrix<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<d_matrix<double>> OR = {{0}, {1}, {1}, {0}};

std::unordered_map<d_matrix<double>, d_matrix<double>> XOR = { {X.at(0), OR.at(0)},
                                                               {X.at(1), OR.at(1)},
                                                               {X.at(2), OR.at(2)},
                                                               {X.at(3), OR.at(3)}};

int main(){
    Adam inputlayer(2, 4, 0.0001, InitType::He);
    ActivateLayer Act1(4, 1, ActivationType::LReLU);
    Adam hiddenlayer1(4, 4, 0.0001, InitType::He);
    ActivateLayer Act2(4, 1, ActivationType::LReLU);
    Adam outputlayer(4, 1, 0.0001, InitType::He);
    ActivateLayer Act3(1, 1, ActivationType::LReLU);
    LossLayer loss(1, 1, LossType::MSE);

    int epochs = 100;
    int sample_count = 4;
    for(int epoch = 1; epoch <= epochs; epoch++){
        double totalloss = 0.00;
        for(const auto& kv : XOR){
            const d_matrix<double> input = kv.first;
            const d_matrix<double> target = kv.second;
            const d_matrix<double> dummy(1, 1);

            inputlayer.feedforward(input);
            Act1.pushInput(inputlayer.getOutput()); Act1.Active();
            hiddenlayer1.feedforward(Act1.getOutput());
            Act2.pushInput(hiddenlayer1.getOutput()); Act2.Active();
            outputlayer.feedforward(Act2.getOutput());
            Act3.pushInput(outputlayer.getOutput()); Act3.Active();
            loss.pushOutput(Act3.getOutput()); loss.pushTarget(target); 
            totalloss += loss.getLoss();

            outputlayer.backprop(nullptr, loss.getGrad(), Act3.d_Active(outputlayer.getOutput()));
            hiddenlayer1.backprop(&outputlayer, dummy, Act2.d_Active(hiddenlayer1.getOutput()));
            inputlayer.backprop(&hiddenlayer1, dummy, Act1.d_Active(inputlayer.getOutput()));
        }
        std::cout << "Epoch" << epoch <<"loss:" << totalloss/sample_count <<std::endl;
    }

    inputlayer.saveWeight();
    Act1.saveLayer();
    hiddenlayer1.saveWeight();
    Act2.saveLayer();
    outputlayer.saveWeight();
    Act3.saveLayer();
    loss.saveLayer();
}
