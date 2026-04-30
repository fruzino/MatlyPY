import numpy as np
import matplotlib.pyplot as plt
import logging
import turtle
from PIL import Image
import cv2

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
    @staticmethod
    def imgplot(data, size=(10, 5), title="Untitled", labels=None, grid=False, cmap=None, **kwargs):
        """
        Displays an image with extensible options.
        :param size: Tuple of (width, height)
        :param labels: Dict containing 'x' and 'y' keys
        :param cmap: Color map for 2D data
        :param kwargs: Additional arguments for plt.imshow()
        """
        labels = labels or {"x": "X", "y": "Y"}
        
        try:
            fig, ax = plt.subplots(figsize=size)

            im = ax.imshow(data, cmap=cmap, **kwargs)

            ax.set_title(title)
            ax.set_xlabel(labels.get("x"))
            ax.set_ylabel(labels.get("y"))
            ax.grid(grid)

            if cmap:
                fig.colorbar(im, ax=ax)
                
            plt.show()
            plt.close(fig)
            
        except Exception as e:
            print(f"Plotting Error: {e}")   
    @staticmethod
    def convert(source, target):
        """
        Convert any file format into another file format
        For example if you have a PNG file and want to change it into a
        NPY file and vice-versa.
        """
        arr = np.load(source)

        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        cv2.imwrite(target, arr)

    @staticmethod
    def theme(ax, theme="dark", title=None, grid=True, hide_spines=True):
        """Apply theme to a plt"""

        if theme == "dark":
            plt.style.use('dark_background')
            ax.set_facecolor('#1e1e1e')
        else:
            plt.style.use('default')

        if hide_spines:
            for s in ['top', 'right']:
                ax.spines[s].set_visible(False)

        if grid:
            ax.grid(True, alpha=0.2, linestyle='--')
        
        if title:
            ax.set_title(title, pad=15, fontweight='bold')

        return ax

    @staticmethod
    def annotate(ax, text, xy, xytext):
        """Add clean callouts to data points."""
        ax.annotate(text, xy=xy, xytext=xytext,
                    arrowprops=dict(arrowstyle='->', color='gray'))

    @staticmethod
    def fill(ax, x, y1, y2=0, color='cyan', alpha=0.2):
        """Add depth with filled backgrounds."""
        ax.fill_between(x, y1, y2, color=color, alpha=alpha)
    class turtlot:
        @staticmethod
        def turtplot(data, speed=3):
            """
            Maps a NumPy array of shape (N, 2) to turtle movements.
            This is not intended for commercial use!!
            """
            if not isinstance(data, np.ndarray):
                raise TypeError("Input must be a NumPy array.")

            screen = turtle.Screen()
            t = turtle.Turtle()
            t.speed(speed)

            t.penup()
            t.goto(data[0, 0], data[0, 1])
            t.pendown()
            
            for point in data[1:]:
                t.goto(point[0], point[1])
            
            screen.mainloop()
        @staticmethod
        def turtplot_img(path, scale=1):
            """
            Renders an image file by mapping pixel data to turtle movements.
            This is not intended for commercial use!!
            """
            img = Image.open(path).convert('RGB')
            width, height = img.size
            img = img.resize((int(width * scale), int(height * scale)))
            
            data = np.array(img)
            rows, cols, _ = data.shape

            screen = turtle.Screen()
            screen.colormode(255)
            screen.tracer(0) 
            
            t = turtle.Turtle()
            t.penup()
            t.speed(0)

            for r in range(rows):
                for c in range(cols):
                    r_val, g_val, b_val = data[r, c]
                    t.goto(c - (cols / 2), (rows / 2) - r)
                    t.dot(2, (r_val, g_val, b_val))

                screen.update()

            screen.mainloop()