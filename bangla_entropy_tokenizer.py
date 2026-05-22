# file: bangla_entropy_tokenizer.py
import json
import pickle
import bangla_entropy_backend

class Trainer:
    @staticmethod
    def train(input_file: str, vocab_size: int = 32000, model_prefix: str = "bangla"):
        engine = bangla_entropy_backend.TokenizerEngine()
        engine.train_from_file(input_file, vocab_size)
        
        try:
            with open("vocab.json", "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            
            model_data = {
                "type": "entropy_bpe_v1",
                "vocab_size": len(vocab_data),
                "vocab": list(vocab_data.values())
            }
            
            model_filename = f"{model_prefix}.model"
            with open(model_filename, "wb") as f_model:
                pickle.dump(model_data, f_model)
                
            print(f"[Success] '{model_filename}' binary model file successfully created!")
            
        except Exception as e:
            print(f"Error while creating .model file: {e}")


class BanglaTokenizer:
    def __init__(self, model_file="bangla.model"):
        with open(model_file, "rb") as f:
            model_data = pickle.load(f)
            
        self.vocab = set(model_data["vocab"])
        print(f"Successfully loaded model from '{model_file}'. Total Vocab: {model_data['vocab_size']}. Runtime Ready!")

    def encode(self, text: str) -> list:
        words = text.split()
        final_tokens = []
        
        for word in words:
            fixed_word = "_" + word
            i = 0
            while i < len(fixed_word):
                match_found = False
                for length in range(min(24, len(fixed_word) - i), 0, -1):
                    sub = fixed_word[i:i+length]
                    if sub in self.vocab:
                        final_tokens.append(sub)
                        i += length
                        match_found = True
                        break
                if not match_found:
                    final_tokens.append(fixed_word[i])
                    i += 1
        return final_tokens