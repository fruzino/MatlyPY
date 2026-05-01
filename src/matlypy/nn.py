import numpy as np
import matplotlib.pyplot as plt
from .mathematics import math as matmath
from .model import model

class nn:
    def __init__(self, vocab):
        self.vocab = vocab
        self.v_size = len(vocab)
        self.W = np.random.randn(self.v_size, self.v_size) * 0.01

    def generate(self, start_word, length=10):
        """
        Generate model output 
        startword
        length
        """
        current_word = start_word
        results = [current_word]
        
        for _ in range(length):
            input_vec = np.zeros(self.v_size)
            if current_word in self.vocab:
                input_vec[self.vocab.index(current_word)] = 1

            next_word, prob = model.predict(input_vec, self.W, self.vocab)
            
            if next_word:
                results.append(next_word)
                current_word = next_word
            else:
                break
        return " ".join(results)