import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pyplot")

class math:
    @staticmethod
    def tensmultiply(tensor1, tensor2):
        try:
            t1, t2 = np.asanyarray(tensor1), np.asanyarray(tensor2)
            if t1.shape[-1] != t2.shape[0]:
                raise ValueError(f"Incompatible shapes for matmul: {t1.shape} and {t2.shape}")
            return np.matmul(t1, t2)
        except ValueError as e:
            logger.error(f"Shape Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Math Error: {e}")
            raise

    @staticmethod
    def tensangle(tensor1, tensor2, degree=True):
        try:
            a, b = np.asanyarray(tensor1), np.asanyarray(tensor2)
            if a.shape != b.shape:
                raise ValueError("Tensors must have the same shape for angle calculation.")
            
            dot = np.dot(a.flatten(), b.flatten())
            norms = np.linalg.norm(a) * np.linalg.norm(b)
            
            if norms == 0:
                return 0.0
                
            cos_theta = np.clip(dot / norms, -1.0, 1.0)
            angle = np.arccos(cos_theta)
            return np.degrees(angle) if degree else angle
        except Exception as e:
            logger.error(f"Error in tensor_angle: {e}")
            return None

    @staticmethod
    def standardize(data):
        try:
            arr = np.asanyarray(data)
            std = np.std(arr)
            if std == 0:
                return arr - np.mean(arr)
            return (arr - np.mean(arr)) / std
        except Exception as e:
            logger.error(f"Error in standardize: {e}")
            return data

    @staticmethod
    def relu(tensor):
        return np.maximum(0, np.asanyarray(tensor))

    @staticmethod
    def softmax(tensor):
        try:
            z = np.asanyarray(tensor)
            if z.size == 0:
                return z
            e = np.exp(z - np.max(z))
            return e / np.sum(e)
        except Exception as e:
            logger.error(f"Error in softmax: {e}")
            return None


