// file: include/trie.h
#pragma once
#include <string>
#include <unordered_map>
#include <memory>

class TrieNode {
public:
    std::unordered_map<char, std::shared_ptr<TrieNode>> children;
    long long count = 0;   
    double loss = 0.0;
    bool is_end = false;
};

class BanglaTrie {
public:
    std::shared_ptr<TrieNode> root;
    BanglaTrie() { root = std::make_shared<TrieNode>(); }

    void insert(const std::string& token, long long initial_count = 1) {
        auto current = root;
        for (char c : token) {
            if (current->children.find(c) == current->children.end()) {
                current->children[c] = std::make_shared<TrieNode>();
            }
            current = current->children[c];
        }
        current->is_end = true;
        current->count += initial_count;
    }

    void set_loss(const std::string& token, double loss_score) {
        auto current = root;
        for (char c : token) {
            if (current->children.find(c) == current->children.end()) {
                return;
            }
            current = current->children[c];
        }
        if (current->is_end) {
            current->loss = loss_score;
        }
    }
};