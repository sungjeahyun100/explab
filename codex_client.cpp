#include <iostream>
#include <cstdlib>
#include <string>
#include <curl/curl.h>
#include "json.hpp"  // https://github.com/nlohmann/json

using json = nlohmann::json;

// libcurl write callback to accumulate response
static size_t write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* s = static_cast<std::string*>(userp);
    size_t total = size * nmemb;
    s->append(static_cast<char*>(contents), total);
    return total;
}

int main() {
    // 1) 환경변수에서 API 키 읽기
    const char* api_key = std::getenv("OPENAI_API_KEY");
    if (!api_key) {
        std::cerr << "ERROR: OPENAI_API_KEY not set" << std::endl;
        return 1;
    }

    // 2) CURL 초기화
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "ERROR: curl_easy_init() failed" << std::endl;
        return 1;
    }

    std::string readBuf;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth = std::string("Authorization: Bearer ") + api_key;
    headers = curl_slist_append(headers, auth.c_str());

    // 3) 요청 바디 구성
    json body = {
        {"model", "code-davinci-002"},
        {"prompt", "// CUDA C++로 벡터 덧셈 함수를 구현해줘\n"},
        {"max_tokens", 128},
        {"temperature", 0.2}
    };
    std::string body_str = body.dump();

    curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/completions");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuf);

    // 4) 요청 실행 및 raw 응답 출력
    CURLcode res = curl_easy_perform(curl);
    std::cout << "=== RAW RESPONSE ===\n" << readBuf << std::endl;
    if (res != CURLE_OK) {
        std::cerr << "curl error: " << curl_easy_strerror(res) << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return 1;
    }

    // 5) JSON 파싱
    json j;
    try {
        j = json::parse(readBuf);
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return 1;
    }

    // 6) API 에러 처리
    if (j.contains("error")) {
        std::cerr << "API Error: "
                  << j["error"]["message"].get<std::string>()
                  << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return 1;
    }

    // 7) choices / text 유효성 검사
    if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty() ||
        !j["choices"][0].contains("text") || j["choices"][0]["text"].is_null()) {
        std::cerr << "No valid 'text' in response." << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return 1;
    }

    // 8) 생성된 코드 출력
    std::cout << "=== GENERATED CODE ===\n"
              << j["choices"][0]["text"].get<std::string>()
              << std::endl;

    // 9) 리소스 정리
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return 0;
}
