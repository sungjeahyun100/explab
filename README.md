# explab

인공지능 및 게임 관련 주제를 실험해보기 위한 코드베이스.

현재 진행중인 실험:
Game of Life의 예측 불가능성을 인공지능에게 풀어보게 시키기

transformer 모듈을 적용한 체스 인공지능 만들기

docker환경 불러오는 명령어:sudo docker run --rm -it --gpus all   -v "$(pwd):/workspace"   -w /workspace   codex-cuda-dev

데이터셋의 라벨은 네 자리 숫자로 저장되며, 학습 네트워크의 출력층 또한 4개의 노드를 사용한다. `LoadingData()` 함수에서 라벨을 각 자리수로 분리하여 사용한다.


