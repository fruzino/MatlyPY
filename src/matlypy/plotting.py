import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PyPlot")

class plot:
    @staticmethod
    def plot(array, height, width, axis=0, title="Untitled", xlabel="X", ylabel="Y", grid=False):
        try:
            arr = np.asanyarray(array)
            if arr.size == 0:
                raise ValueError("Array is empty.")
            
            plt.figure(figsize=(width, height))
            plt.plot(np.mean(arr, axis=axis))
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(grid)
            plt.show()
            plt.close()
        except ValueError as e:
            logger.error(f"Value Error in plot_mean: {e}")
        except Exception as e:
            logger.error(f"Unexpected Error in plot_mean: {e}")

    @staticmethod
    def autoplot(array, title="Untitled", xlabel="X", ylabel="Y", grid=False):
        try:
            arr = np.asanyarray(array)
            if arr.size == 0:
                raise ValueError("Array is empty.")
                
            plot_axis = 0 if arr.ndim > 1 else None
            plt.figure(figsize=(10, 6))
            plt.plot(np.mean(arr, axis=plot_axis))
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(grid)
            plt.show()
            plt.close()
        except Exception as e:
            logger.error(f"Error in plot_auto: {e}")
