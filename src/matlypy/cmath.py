import ctypes
import numpy as np
import os

class cmath:
    def __init__(self, dll_name="dll_cmath.dll"):
        _path = os.path.join(os.path.dirname(__file__), dll_name)
        if not os.path.exists(_path):
            raise FileNotFoundError(f"Could not find {dll_name} at {_path}")
        
        self._lib = ctypes.CDLL(_path)
        self._setup_types()

    def _setup_types(self):
        """Map Python types to C function signatures."""

        double_ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')

        self._lib.mat_mul.argtypes = [double_ptr, double_ptr, double_ptr, 
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._lib.mat_mul.restype = None

        self._lib.standardize.argtypes = [double_ptr, double_ptr, ctypes.c_int]
        self._lib.standardize.restype = None

        self._lib.relu.argtypes = [double_ptr, double_ptr, ctypes.c_int]
        self._lib.relu.restype = None

        self._lib.softmax.argtypes = [double_ptr, double_ptr, ctypes.c_int]
        self._lib.softmax.restype = None

        self._lib.tensangle.argtypes = [double_ptr, double_ptr, ctypes.c_int]
        self._lib.tensangle.restype = ctypes.c_double

    def tensmultiply(self, tensor1, tensor2):
        """High-speed matrix multiplication."""
        t1 = np.ascontiguousarray(tensor1, dtype=np.float64)
        t2 = np.ascontiguousarray(tensor2, dtype=np.float64)
        
        if t1.ndim != 2 or t2.ndim != 2:
            raise ValueError("Inputs must be 2D matrices.")
        if t1.shape[1] != t2.shape[0]:
            raise ValueError(f"Shape mismatch: {t1.shape} and {t2.shape}")

        m, n = t1.shape
        p = t2.shape[1]
        res = np.zeros((m, p), dtype=np.float64)
        
        self._lib.mat_mul(t1, t2, res, m, n, p)
        return res

    def standardize(self, data):
        """Z-score normalization (Mean 0, Std 1)."""
        arr = np.ascontiguousarray(data, dtype=np.float64)
        out = np.zeros_like(arr)
        self._lib.standardize(arr, out, arr.size)
        return out

    def relu(self, tensor):
        """Rectified Linear Unit activation."""
        arr = np.ascontiguousarray(tensor, dtype=np.float64)
        out = np.zeros_like(arr)
        self._lib.relu(arr, out, arr.size)
        return out

    def softmax(self, tensor):
        """Softmax activation for probability distributions."""
        arr = np.ascontiguousarray(tensor, dtype=np.float64)
        out = np.zeros_like(arr)
        self._lib.softmax(arr, out, arr.size)
        return out

    def tensangle(self, tensor1, tensor2):
        """Calculate angle between two tensors in degrees."""
        t1 = np.ascontiguousarray(tensor1, dtype=np.float64)
        t2 = np.ascontiguousarray(tensor2, dtype=np.float64)
        if t1.shape != t2.shape:
            raise ValueError("Tensors must have the same shape.")
            
        return self._lib.tensangle(t1, t2, t1.size)

