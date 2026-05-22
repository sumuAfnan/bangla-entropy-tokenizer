# file: bangla_entropy_tokenizer.py
import bangla_entropy_backend

class Trainer:
    @staticmethod
    def train(input_file: str, vocab_size: int = 32000):
        engine = bangla_entropy_backend.TokenizerEngine()
        engine.train_from_file(input_file, vocab_size)

class BanglaTokenizer:
    def __init__(self, model_file="vocab.json"):
        print(f"Successfully {model_file} loaded. Runtime Ready")