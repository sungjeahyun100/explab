#include "chess.hpp"

int main(){
    chessboard test1(positionType::DEFAULT);
    test1.printBoard();
    debugChessboardvar(&test1);
    PGN Default_Move_e4 = {pieceColor::WHITE, pieceType::PAWN, 4, 6, 4, 4, pieceType::NONE};
    test1.updateBoard(Default_Move_e4);
    test1.printBoard();
    debugChessboardvar(&test1);
    return 0;
}