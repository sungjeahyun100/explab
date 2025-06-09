#include "perceptron.hpp"

// 현재 시간 문자열 반환 (가중치 저장 파일명에 사용)
std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now = *std::localtime(&t_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d_%H%M%S");
    return oss.str();
}

// 가중치 파일에서 weight, bias 불러오기
void perceptronLayer::loadWeight(const std::string &path)
{
    std::ifstream test_subject(path, std::ios::binary);
    if (!test_subject) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }

    test_subject >> weight;
    test_subject >> bias;

    test_subject.close();
}

// 가중치 파일로 저장 (이름: subject+타임스탬프)
void perceptronLayer::saveWeight()
{
    std::ofstream test_subject(WEIGHT_DATAPATH + "subject" + getCurrentTimestamp() + ".bin");
    test_subject << weight;
    test_subject << bias;
    test_subject.close();
}

d_matrix<double>& perceptronLayer::getOutput() { return output; }

// weight, bias를 GPU로 복사
void perceptronLayer::updateWeightInDev() {
    weight.cpyToDev();
    bias.cpyToDev();
}

// feedforward: z = W x + b, output = z
// (활성화는 ActivateLayer에서 적용)
void perceptronLayer::feedforward(const d_matrix<double>& raw_input) {
    input = raw_input;

    z = matrixPlus(matrixMP(weight, input), bias);

    output = z;
}

// 그래디언트 계산 (델타, Gt_W, Gt_B)
// 델타: δ = (next->weight^T * next->delta) ⊙ act_deriv
// Gt_W = δ * input^T, Gt_B = δ
void perceptronLayer::calculateGrad(perceptronLayer* next, const d_matrix<double>& external_delta, const d_matrix<double>& act_deriv) {
    d_matrix<double> grad_input = external_delta;

    if (next != nullptr) {
        d_matrix<double> weighted_delta = matrixMP(next->weight.transpose(), next->delta);
        weighted_delta.cpyToDev();
        grad_input = weighted_delta;
    }

    delta = HadamardProduct(grad_input, act_deriv);
    delta.cpyToDev();

    Gt_W = matrixMP<double>(delta, input.transpose());
    Gt_B = delta;

    cudaDeviceSynchronize();
}

// 입력 설정 (input = in)
void ActivateLayer::pushInput(const d_matrix<double>& in){
    input = in;
    input.cpyToDev();
}

// 활성화 적용 (output = f(input))
// 지원: ReLU, LReLU, Identity, Sigmoid
void ActivateLayer::Active(){
    switch (act) {
        case ActivationType::ReLU:
            output = MatrixActivate<double, relu>(input); break;
        case ActivationType::LReLU:
            output = MatrixActivate<double, lrelu>(input); break;
        case ActivationType::Identity:
            output = MatrixActivate<double, Identity>(input); break;
        case ActivationType::Sigmoid:
            output = MatrixActivate<double, sigmoid>(input); break;
        case ActivationType::Tanh:
            output = MatrixActivate<double, Tanh>(input); break;
        default:
            throw std::runtime_error("Unsupported ActivationType in perceptronLayer");
    }
}

// 활성화 함수 미분값 반환 (f'(z))
// ReLU: 1(x>0), 0(x<=0)
// LReLU: 1(x>0), 0.01(x<=0)
// Identity: 1
// Sigmoid: σ'(x) = σ(x)(1-σ(x))
d_matrix<double> ActivateLayer::d_Active(const d_matrix<double>& z) {
    switch (act) {
        case ActivationType::ReLU:
            return MatrixActivate<double, d_relu>(z);
        case ActivationType::LReLU:
            return MatrixActivate<double, d_lrelu>(z);
        case ActivationType::Identity:
            return MatrixActivate<double, d_I>(z);
        case ActivationType::Sigmoid:
            return MatrixActivate<double, d_sigmoid>(z);
        case ActivationType::Tanh:
            return MatrixActivate<double, d_tanh>(z);
        default:
            throw std::runtime_error("Unsupported ActivationType in d_Active");
    }
}

// 활성화 결과 반환
const d_matrix<double>& ActivateLayer::getOutput() const {
    return output; 
}

// 타겟 입력
void LossLayer::pushTarget(const d_matrix<double>& Target){
    target = Target;
}

// 출력 입력
void LossLayer::pushOutput(const d_matrix<double>& Output){
    output = Output;
}

// 손실값 반환
// MSE: L = 1/n Σ(y-p)^2
// CrossEntropy: L = -Σ y log(softmax(p))
double LossLayer::getLoss(){
    // 1) 디바이스→호스트 복사
    output.cpyToHost();
    target.cpyToHost();

    switch (Loss)
    {
        case LossType::MSE: {
            // MSE: L = 1/N Σ (output − target)², 전부 호스트 계산
            int N = output.getRow();
            double sum = 0.0;
            for (int i = 0; i < N; ++i) {
                double diff = output(i, 0) - target(i, 0);
                sum += diff * diff;
            }
            return sum / static_cast<double>(N);
        }

        case LossType::CrossEntropy: {
            // 이 구현은 “이진 크로스엔트로피” (비트 단위 분류) 예시입니다.
            // 필요하다면 multi-class softmax 버전으로 바꾸시면 됩니다.
            int N = output.getRow();
            double loss = 0.0;
            for (int i = 0; i < N; ++i) {
                double z = output(i, 0);
                double y = target(i, 0);
                // sigmoid로 확률 p 계산
                double p = 1.0 / (1.0 + std::exp(-z));
                // log(0) 방지용 클리핑
                p = std::min(std::max(p, 1e-7), 1.0 - 1e-7);
                loss += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
            }
            return loss;
        }

        default:
            throw std::runtime_error("Unsupported LossType in getLoss");
    }
}

// 손실 미분 반환
// MSE: dL/dz = 2(y-p)
// CrossEntropy: dL/dz = softmax(p) - y
d_matrix<double> LossLayer::getGrad() {
    // 1) 디바이스→호스트 복사
    output.cpyToHost();
    target.cpyToHost();

    switch (Loss) {
        case LossType::MSE: {
            // L = (1/N) Σ (o - t)^2  이므로  dL/dz = 2*(o - t)/N
            int N = output.getRow();
            // diff = output - target
            d_matrix<double> diff = matrixPlus(output, ScalaProduct(target, -1.0));
            return ScalaProduct(diff, 2.0 / static_cast<double>(N));
        }

        case LossType::CrossEntropy: {
            // 이진 크로스엔트로피 (BCE) + Sigmoid 로 구현
            int N = output.getRow();
            d_matrix<double> grad(N, 1);
            for (int i = 0; i < N; ++i) {
                double z = output(i, 0);
                double y = target(i, 0);
                // Sigmoid 확률
                double p = 1.0 / (1.0 + std::exp(-z));
                // gradient = p - y
                grad(i, 0) = p - y;
            }
            return grad;
        }

        default:
            throw std::runtime_error("Unsupported LossType in getGrad");
    }
}
Adam::~Adam(){}

// Adam 옵티마이저 역전파
// m, v: 1차/2차 모멘트, 베타1/2, epsilon, t(스텝)
// 업데이트 수식:
// m = β₁ m + (1-β₁)g, v = β₂ v + (1-β₂)g²
// m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)
// W -= lr * m̂/(sqrt(v̂)+ε)
void Adam::backprop(perceptronLayer* next, const d_matrix<double>& external_delta, const d_matrix<double>& act_deriv){

    this->t++;

    // 1) gradient 계산
    this->calculateGrad(next, external_delta, act_deriv);

    // 2) 1차 및 2차 모멘트 갱신
    this->m_W = matrixPlus(ScalaProduct(this->m_W, this->beta1), ScalaProduct(this->Gt_W, 1.0 - this->beta1));
    this->v_W = matrixPlus(ScalaProduct(this->v_W, this->beta2), ScalaProduct(HadamardProduct(this->Gt_W, this->Gt_W), 1.0 - this->beta2));
    this->m_B = matrixPlus(ScalaProduct(this->m_B, this->beta1), ScalaProduct(this->Gt_B, 1.0 - this->beta1));
    this->v_B = matrixPlus(ScalaProduct(this->v_B, this->beta2), ScalaProduct(HadamardProduct(this->Gt_B, this->Gt_B), 1.0 - this->beta2));

    // 3) 편향 보정 계수
    double bias_corr1 = 1.0 - std::pow(this->beta1, this->t);
    double bias_corr2 = 1.0 - std::pow(this->beta2, this->t);

    // 4) 편향 보정된 모멘트
    d_matrix<double> m_W_hat = ScalaProduct(this->m_W, 1.0 / bias_corr1);
    d_matrix<double> v_W_hat = ScalaProduct(this->v_W, 1.0 / bias_corr2);
    d_matrix<double> m_B_hat = ScalaProduct(this->m_B, 1.0 / bias_corr1);
    d_matrix<double> v_B_hat = ScalaProduct(this->v_B, 1.0 / bias_corr2);

    // 5) 분모: sqrt(v̂) + ε
    //    MatrixActivate<sqrt> 는 elementwise sqrt, devide 는 reciprocal
    auto sqrt_vW = MatrixActivate<double, sqr>(v_W_hat);
    auto denomW  = ScalaPlus(sqrt_vW, this->epsilon);
    auto invDenW = MatrixActivate<double, devide>(denomW);

    auto sqrt_vB = MatrixActivate<double, sqr>(v_B_hat);
    auto denomB  = ScalaPlus(sqrt_vB, this->epsilon);
    auto invDenB = MatrixActivate<double, devide>(denomB);

    // 6) 파라미터 업데이트
    //    w ← w − lr * (m̂ ⊙ invDen)
    this->weight = matrixPlus(
        this->weight,
        ScalaProduct(HadamardProduct(m_W_hat, invDenW), -this->learning_rate)
    );
    this->bias = matrixPlus(
        this->bias,
        ScalaProduct(HadamardProduct(m_B_hat, invDenB), -this->learning_rate)
    );

    // 7) 디바이스 메모리에 복사
    this->updateWeightInDev();
    cudaDeviceSynchronize();

}

SGD::~SGD(){}

// SGD 옵티마이저 역전파
// W -= lr * grad
void SGD::backprop(perceptronLayer* next, const d_matrix<double>& external_delta, const d_matrix<double>& act_deriv)
{
    this->calculateGrad(next, external_delta, act_deriv);
    this->weight = matrixPlus(this->weight, ScalaProduct(this->Gt_W, (-1) * this->learning_rate));
    this->bias = matrixPlus(this->bias, ScalaProduct(this->Gt_B, (-1) * this->learning_rate));
    this->updateWeightInDev();
    cudaDeviceSynchronize();
}

/*
[MLP(다층 퍼셉트론) 구성 예시]

// 1. 계층 선언 (입력, 은닉, 출력)
SGD input_layer(입크기, 은닉크기, lr, InitType::He);
ActivateLayer act1(은닉크기, 1, ActivationType::ReLU);
SGD output_layer(은닉크기, 출력크기, lr, InitType::He);
ActivateLayer act2(출력크기, 1, ActivationType::Sigmoid); // 또는 Softmax
LossLayer loss(출력크기, 1, LossType::CrossEntropy);

// 2. 순전파 예시
input_layer.feedforward(input); // 첫 계층
act1.pushInput(input_layer.getOutput());
act1.Active();
output_layer.feedforward(act1.getOutput());
act2.pushInput(output_layer.getOutput());
act2.Active();

// 3. 역전파 예시
loss.pushTarget(target);
loss.pushOutput(act2.getOutput());
d_matrix<double> grad = loss.getGrad();
output_layer.backprop(nullptr, grad, act2.d_Active(output_layer.getOutput()));
input_layer.backprop(&output_layer, output_layer.delta, act1.d_Active(input_layer.getOutput()));

[MLP(다층 퍼셉트론) 구성 예시 - Adam 사용]

// 1. 계층 선언 (입력, 은닉, 출력)
Adam input_layer(입크기, 은닉크기, lr, InitType::He);
ActivateLayer act1(은닉크기, 1, ActivationType::ReLU);
Adam output_layer(은닉크기, 출력크기, lr, InitType::He);
ActivateLayer act2(출력크기, 1, ActivationType::Sigmoid); // 또는 Softmax
LossLayer loss(출력크기, 1, LossType::CrossEntropy);

// 2. 순전파 예시
input_layer.feedforward(input); // 첫 계층
act1.pushInput(input_layer.getOutput());
act1.Active();
output_layer.feedforward(act1.getOutput());
act2.pushInput(output_layer.getOutput());
act2.Active();

// 3. 역전파 예시
loss.pushTarget(target);
loss.pushOutput(act2.getOutput());
d_matrix<double> grad = loss.getGrad();
output_layer.backprop(nullptr, grad, act2.d_Active(output_layer.getOutput()));
input_layer.backprop(&output_layer, output_layer.delta, act1.d_Active(input_layer.getOutput()));
*/



