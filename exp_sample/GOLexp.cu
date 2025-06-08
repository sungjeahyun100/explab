#include "database.hpp"
#include "perceptron.hpp"

Adam inputlayer(100, 256, 0.001, InitType::He);
ActivateLayer input(256, 1, ActivationType::LReLU);
Adam hiddenlayer1(256, 256, 0.001, InitType::He);
ActivateLayer hidden1(256, 1, ActivationType::LReLU);
Adam outputlayer(256, BOARDHEIGHT*BOARDWIDTH, 0.001, InitType::He);
ActivateLayer output(BOARDHEIGHT*BOARDWIDTH, 1, ActivationType::LReLU);
LossLayer loss(BOARDHEIGHT*BOARDWIDTH, 1, LossType::MSE);



int main(){
    
}