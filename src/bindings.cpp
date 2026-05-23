// file: src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include "tokenizer_engine.cpp"

namespace py = pybind11;

class PythonTokenizerWrapper {
private:
    TokenizerEngine engine;
    std::vector<std::string> tokens_sorted;

    std::string trim_quotes(const std::string& str) {
        size_t first = str.find_first_of("\"");
        size_t last = str.find_last_of("\"");
        if (first != std::string::npos && last != std::string::npos && first < last) {
            return str.substr(first + 1, last - first - 1);
        }
        return str;
    }

public:
    void train(const std::string& file_path, int vocab_size) {
        engine.train_from_file(file_path, vocab_size);
    }

    void load_model(const std::string& model_path) {
        tokens_sorted.clear();
        std::ifstream file(model_path);
        if (!file.is_open()) return;

        std::string line;
        while (std::getline(file, line)) {
            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) continue;

            std::string token_part = line.substr(colon_pos + 1);
            std::string token = trim_quotes(token_part);
            
            if (!token.empty() && token != "{" && token != "}") {
                if (token.back() == ',') token.pop_back();
                token = trim_quotes(token);
                if (!token.empty()) {
                    tokens_sorted.push_back(token);
                }
            }
        }

        std::sort(tokens_sorted.begin(), tokens_sorted.end(), [](const std::string& a, const std::string& b) {
            return a.length() > b.length();
        });
    }

    std::vector<std::string> encode(const std::string& text) {
        std::string normalized = "";
        bool last_was_space = true; 

        for (size_t i = 0; i < text.length(); ++i) {
            if (text[i] == ' ') {
                if (!last_was_space) {
                    normalized += "_";
                    last_was_space = true;
                }
            } else {
                if (last_was_space) {
                    if (normalized.empty() || normalized.back() != '_') {
                        normalized += "_";
                    }
                    last_was_space = false;
                }
                normalized += text[i];
            }
        }

        size_t n = normalized.length();
        std::vector<int> dp(n + 1, 1e9); 
        std::vector<int> parent(n + 1, -1);
        std::vector<std::string> matched_token(n + 1, "");

        dp[0] = 0;

        for (size_t i = 0; i < n; ++i) {
            if (dp[i] == 1e9) continue;

            bool any_match = false;
            for (const auto& token : tokens_sorted) {
                size_t len = token.length();
                if (i + len <= n && normalized.compare(i, len, token) == 0) {
                    any_match = true;
                    if (dp[i] + 1 < dp[i + len]) {
                        dp[i + len] = dp[i] + 1;
                        parent[i + len] = i;
                        matched_token[i + len] = token;
                    }
                }
            }

            if (!any_match || true) { 
                size_t len = 1;
                unsigned char c = normalized[i];
                if (c >= 0xf0) len = 4;
                else if (c >= 0xe0) len = 3;
                else if (c >= 0xc0) len = 2;

                if (i + len <= n && dp[i] + 1 < dp[i + len]) {
                    dp[i + len] = dp[i] + 1;
                    parent[i + len] = i;
                    matched_token[i + len] = normalized.substr(i, len);
                }
            }
        }

        std::vector<std::string> final_tokens;
        int curr = n;
        while (curr > 0 && parent[curr] != -1) {
            final_tokens.push_back(matched_token[curr]);
            curr = parent[curr];
        }
        std::reverse(final_tokens.begin(), final_tokens.end());

        return final_tokens;
    }
};

PYBIND11_MODULE(bangla_entropy_backend, m) {
    py::class_<PythonTokenizerWrapper>(m, "TokenizerEngine")
        .def(py::init<>())
        .def("train", &PythonTokenizerWrapper::train)
        .def("load_model", &PythonTokenizerWrapper::load_model)
        .def("encode", &PythonTokenizerWrapper::encode);
}