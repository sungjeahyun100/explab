#include "perceptron.hpp"
#include "chess.hpp"

int main() {
    int turn = 1;
    chessboard testboard(positionType::TEST_BLACK_CHECK);

    while (true) {
        // (1) 보드와 공격지점, 합법 움직임 출력
        testboard.printBoard();
        debugChessboardvar(&testboard);
        std::cout << "=== Turn " << turn << " ===\n";

        // (2) 흑/백 결정
        std::string player_color_str;
        if (turn % 2 == 1) {
            player_color_str = "w";  // 백
        } else {
            player_color_str = "b";  // 흑
        }

        // (3) 합법 수 갱신 및 출력 (선택 사항)
        testboard.genLegalMoves();
        const auto& legal = testboard.getMoveList();
        // --- 합법 수들을 간략하게 한 줄로 출력하고 싶으면 여기에 추가
        //    ex) for (auto& mv : legal) std::cout << translateMove(mv) << " ";
        //    std::cout << "\n";

        // (4) 사용자 입력
        std::string player_Move;
        std::cout << "(예: e2e4 / Ke1d2 / O-O / O-O-O / quit): ";
        std::cin >> player_Move;
        if (!std::cin || player_Move == "quit" || player_Move == "exit") {
            std::cout << "게임을 종료합니다.\n";
            break;
        }

        // (5) 컬러 + 실제 이동 합치기
        std::string fullPGN = player_color_str + player_Move;

        // (6) 최소 길이 검증 (예외 방지용)
        if (fullPGN.size() < 3) {
            std::cerr << "[오류] 입력 길이가 너무 짧습니다: " 
                      << fullPGN << "\n";
            continue;
        }

        // (7) PGN parsing
        PGN playerMove = translatePGN(fullPGN);

        // (8) parsing 결과 검증
        if (playerMove.color == pieceColor::NONE || playerMove.type == pieceType::NONE) {
            std::cerr << "[오류] 잘못된 PGN 형식: " << fullPGN << "\n";
            continue;
        }
        if (playerMove.fromFile < 0 || playerMove.fromFile > 7 ||
            playerMove.fromRank < 0 || playerMove.fromRank > 7 ||
            playerMove.toFile   < 0 || playerMove.toFile   > 7 ||
            playerMove.toRank   < 0 || playerMove.toRank   > 7) 
        {
            std::cerr << "[오류] 좌표 범위를 벗어났습니다: " << fullPGN << "\n";
            continue;
        }

        // (9) “합법적인 수인지” 검사
        auto it = std::find(legal.begin(), legal.end(), playerMove);
        if (it == legal.end()) {
            std::cerr << "[오류] 비합법적인 수입니다: " << fullPGN << "\n";
            continue;
        }

        // (10) 실제 보드 업데이트
        testboard.updateBoard(playerMove);
        turn++;
    } // end while

    return 0;
}



