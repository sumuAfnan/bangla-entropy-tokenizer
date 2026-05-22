// file: src/tokenizer_engine.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <queue> 
#include <algorithm>
#include "../include/trie.h"

class TokenizerEngine {
private:
    BanglaTrie main_trie;
    long long total_token_count = 0;
    std::unordered_map<std::string, long long> vocab_counts;

    std::vector<std::string> split_utf8(const std::string& str) {
        std::vector<std::string> chars;
        for (size_t i = 0; i < str.length();) {
            size_t len = 1;
            unsigned char c = str[i];
            if (c >= 0xf0) len = 4;
            else if (c >= 0xe0) len = 3; 
            else if (c >= 0xc0) len = 2;
            chars.push_back(str.substr(i, len));
            i += len;
        }
        return chars;
    }

    size_t get_utf8_char_count(const std::string& str) {
        size_t count = 0;
        for (size_t i = 0; i < str.length();) {
            unsigned char c = str[i];
            if (c >= 0xf0) i += 4;
            else if (c >= 0xe0) i += 3;
            else if (c >= 0xc0) i += 2;
            else i += 1;
            count++;
        }
        return count;
    }

public:
    void train_from_file(const std::string& file_path, int vocab_size) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Error: Training file not found!" << std::endl;
            return;
        }

        std::string word;
        std::cout << "[Step 1] Collecting candidate tokens (Character-Level Logic)..." << std::endl;
        
        while (file >> word) {
            std::string fixed_word = "_" + word;
            std::vector<std::string> utf8_chars = split_utf8(fixed_word);
            
            for (size_t i = 0; i < utf8_chars.size(); ++i) {
                std::string sub = "";
                for (size_t len = 1; len <= 24 && i + len <= utf8_chars.size(); ++len) {
                    sub += utf8_chars[i + len - 1];
                    
                    main_trie.insert(sub);
                    vocab_counts[sub]++;
                    total_token_count++;
                }
            }
        }
        file.close();

        std::cout << "[Step 2] Scoring tokens with Greedy Length Bonus..." << std::endl;
        std::unordered_map<std::string, double> vocab_loss;
        
        using TokenPair = std::pair<double, std::string>;
        std::priority_queue<TokenPair> pq;

        for (auto& [token, count] : vocab_counts) {
            double probability = (double)count / total_token_count;
            double entropy = -std::log2(probability);
            
            size_t char_len = get_utf8_char_count(token);
            double score = entropy;
            
            if (token.rfind("_", 0) == 0) { 
                if (char_len >= 4 && char_len <= 15) {        
                    score -= 18.0; 
                } else if (char_len > 15) {
                    score += 5.0; 
                } else {
                    score -= 2.0; 
                }
            } else {                        
                if (token == "গুলো" || token == "দের" || token == "রা" || token == "টি") {
                    score -= 12.0; 
                } else if (char_len <= 3) {
                    score -= 4.0;  
                } else {
                    score += 25.0; 
                }
            }

            vocab_loss[token] = score;
            main_trie.set_loss(token, score);
            pq.push({score, token}); 
        }

        std::cout << "[Step 3] Pruning vocabulary..." << std::endl;
        while (vocab_counts.size() > (size_t)vocab_size && !pq.empty()) {
            TokenPair top_node = pq.top();
            std::string worst_token = top_node.second;
            pq.pop();

            if (vocab_counts.find(worst_token) != vocab_counts.end()) {
                vocab_counts.erase(worst_token);
                vocab_loss.erase(worst_token);
            }
        }

        std::cout << "[Step 4] Training complete. Saved final " << vocab_size << " optimized tokens." << std::endl;
        save_to_json(vocab_counts);
    }

private:
    void save_to_json(const std::unordered_map<std::string, long long>& final_vocab) {
        std::ofstream json_file("vocab.json");
        json_file << "{\n";
        size_t i = 0;
        for (auto& [token, count] : final_vocab) {
            json_file << "  \"" << i << "\": \"" << token << "\"";
            if (i < final_vocab.size() - 1) json_file << ",\n";
            i++;
        }
        json_file << "\n}";
        json_file.close();
    }
};