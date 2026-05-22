// file: src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>
#include <algorithm>
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
        for (size_t i = 0; i < text.length(); ++i) {
            if (text[i] == ' ') {
                normalized += "_";
            } else {
                if (i == 0) normalized += "_";
                normalized += text[i];
            }
        }

        std::vector<std::string> final_tokens;
        size_t i = 0;
        while (i < normalized.length()) {
            bool matched = false;
            for (const auto& token : tokens_sorted) {
                if (normalized.compare(i, token.length(), token) == 0) {
                    final_tokens.push_back(token);
                    i += token.length();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                size_t len = 1;
                unsigned char c = normalized[i];
                if (c >= 0xf0) len = 4;
                else if (c >= 0xe0) len = 3;
                else if (c >= 0xc0) len = 2;
                
                final_tokens.push_back(normalized.substr(i, len));
                i += len;
            }
        }
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