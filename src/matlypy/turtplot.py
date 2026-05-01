import numpy as np
import matplotlib.pyplot as plt
import logging
import turtle
from PIL import Image
import cv2
import librosa
import librosa.display

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
    