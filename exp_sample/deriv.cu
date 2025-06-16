#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

int main() {
    const std::string inputDir  = "../graph/count_ver_loss";
    const std::string outputDir = "../graph/count_ver_loss_deriv";

    // 출력 디렉터리 생성
    try {
        std::filesystem::create_directories(outputDir);
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Directory creation failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // 입력 디렉터리 순회
    for (const auto &entry : std::filesystem::directory_iterator(inputDir)) {
        if (!entry.is_regular_file()) continue;
        auto inPath = entry.path();
        if (inPath.extension() != ".txt") continue;

        // 출력 파일명
        std::string outName = "deriv_" + inPath.filename().string();
        std::filesystem::path outPath = std::filesystem::path(outputDir) / outName;

        std::ifstream fin(inPath);
        std::ofstream fout(outPath);
        if (!fin.is_open() || !fout.is_open()) {
            std::cerr << "Failed to open input or output: "
                      << inPath << " or " << outPath << std::endl;
            continue;
        }

        std::string line;
        double epoch = 0.0, loss = 0.0;
        double prev_loss = 0.0;
        bool first = true;

        // 데이터 라인별 읽기, 주석(#) 및 빈 라인 무시
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            if (!(ss >> epoch >> loss)) continue;

            double dloss = first ? 0.0 : (loss - prev_loss);
            fout << epoch << ' ' << dloss << '\n';

            prev_loss = loss;
            first = false;
        }

        fin.close();
        fout.close();

        // 파일 권한 설정: 읽기/쓰기 추가
        std::error_code ec;
        std::filesystem::permissions(outPath,
            std::filesystem::perms::owner_read  | std::filesystem::perms::owner_write  |
            std::filesystem::perms::group_read  | std::filesystem::perms::group_write  |
            std::filesystem::perms::others_read | std::filesystem::perms::others_write,
            std::filesystem::perm_options::add, ec);
        if (ec) {
            std::cerr << "Permission change failed for " << outName
                      << ": " << ec.message() << std::endl;
        }

        std::cout << "Processed: " << inPath.filename()
                  << " -> " << outName << std::endl;
    }

    return 0;
}
