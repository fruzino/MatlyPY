import numpy as np
import matplotlib.pyplot as plt
import logging
from .mathematics import matmath
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PyPlot")

class model:
    @staticmethod
    def predict(input_vec, weights, labels):
        try:
            scores = matmath.tensor_multiply(input_vec, weights)
            probs = matmath.softmax(scores)
            idx = np.argmax(probs)
            return labels[idx], probs[idx]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, 0.0

    @staticmethod
    def gethot(tag, tags):
        try:
            target = np.zeros(len(tags))
            target[tags.index(tag)] = 1
            return target
        except ValueError:
            logger.error(f"Tag '{tag}' not found in tags list.")
            return None

    @staticmethod
    def weights(weights, inputs, target, output, rate=0.01):
        try:
            error = output - target
            return weights - (rate * np.outer(inputs, error))
        except Exception as e:
            logger.error(f"Weight update failed: {e}")
            return weights

    @staticmethod
    def model(type=1, instruct="", dataset=None, token=5, n=3, data="", temp=1.0):
        """
        N-Gram Model for MatPy.
        n: The 'N' in N-gram (3 = Trigram, 2 = Bigram, etc.)
        """

        if dataset:
            with open(dataset, 'r') as f: data = f.read()
        
        raw_text = f"{instruct} {data}" if instruct else data
        parts = raw_text.split() if type == 1 else list(raw_text)
        
        brain = {}
        context_size = n - 1

        for i in range(len(parts) - context_size):
            context = tuple(parts[i : i + context_size])
            target = parts[i + context_size]
            
            if context not in brain:
                brain[context] = []
            brain[context].append(target)

        seed = instruct.split() if type == 1 else list(instruct)

        if len(seed) < context_size:
            current_context = tuple(parts[:context_size])
        else:
            current_context = tuple(seed[-context_size:])
        
        res = list(seed)
        
        for _ in range(token):
            if current_context not in brain:
                break
            
            possibilities = brain[current_context]

            if temp <= 0.1:
                next_token = max(set(possibilities), key=possibilities.count)
            else:
                tokens, counts = np.unique(possibilities, return_counts=True)
                probs = counts / counts.sum()
                
                logits = np.log(probs + 1e-10) / temp
                exp_logits = np.exp(logits)
                p_temp = exp_logits / np.sum(exp_logits)
                
                next_token = np.random.choice(tokens, p=p_temp)

            res.append(next_token)
            current_context = tuple(res[-context_size:])
            
        joiner = " " if type == 1 else ""
        return joiner.join(res), brain, list(set(parts))
    
    @staticmethod
    def save(filepath, weights, vocab):
        """
        Saves the model weights and vocabulary into a .gguf binary file.
        """
        try:
            with open(filepath, 'wb') as f:
                f.write(struct.pack('<I', 0x46554747)) 

                f.write(struct.pack('<I', 3))

                f.write(struct.pack('<Q', 1)) 
                f.write(struct.pack('<Q', 1)) 

                key = "general.name"
                f.write(struct.pack('<Q', len(key)))
                f.write(key.encode('utf-8'))
                val = "MatPy-Bigram"
                f.write(struct.pack('<Q', len(val)))
                f.write(val.encode('utf-8'))

                f.write(struct.pack('<Q', len(vocab)))
                for word in vocab:
                    encoded_word = word.encode('utf-8')
                    f.write(struct.pack('<Q', len(encoded_word)))
                    f.write(encoded_word)

                w_data = weights.astype(np.float32).tobytes()
                f.write(struct.pack('<Q', len(w_data)))
                f.write(w_data)

            print(f"Brain successfully saved to {filepath}")
        except Exception as e:
            print(f"Failed to save GGUF: {e}")
    @staticmethod
    def load(filepath):
        """
        Loads the model weights and vocabulary from a .gguf binary file.
        Returns: (weights, vocab)
        """
        try:
            with open(filepath, 'rb') as f:
                magic = struct.unpack('<I', f.read(4))[0]
                if magic != 0x46554747:
                    raise ValueError("Not a valid GGUF file.")

                version = struct.unpack('<I', f.read(4))[0]
                
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_count = struct.unpack('<Q', f.read(8))[0]

                key_len = struct.unpack('<Q', f.read(8))[0]
                f.read(key_len)
                val_len = struct.unpack('<Q', f.read(8))[0]
                f.read(val_len) 

                vocab_len = struct.unpack('<Q', f.read(8))[0]
                vocab = []
                for _ in range(vocab_len):
                    word_len = struct.unpack('<Q', f.read(8))[0]
                    vocab.append(f.read(word_len).decode('utf-8'))

                w_data_len = struct.unpack('<Q', f.read(8))[0]
                w_bytes = f.read(w_data_len)

                weights = np.frombuffer(w_bytes, dtype=np.float32).reshape(len(vocab), len(vocab))

            print(f"Brain loaded successfully from {filepath}")
            return weights, vocab
        except Exception as e:
            print(f"Failed to load GGUF: {e}")
            return None, None