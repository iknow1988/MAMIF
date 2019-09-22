import random

import numpy as np
from kivy.graphics import Color, Rectangle
from kivy.uix.widget import Widget

from constants import MARGIN_NO_OBSTICLE, NUM_OBSTABLES


class ObstacleWidget(Widget):

    def __init__(self, width, height, **kwargs):
        # make sure we aren't overriding any important functionality
        super(ObstacleWidget, self).__init__(**kwargs)
        # self.sand = np.zeros((width, height))
        # self.sand = np.random.randint(0, 2, size=(width, height))

        self.width = width
        self.height = height
        self.sand = np.zeros((width, height))

    def load(self):
        rectangles = []
        for _ in range(NUM_OBSTABLES):
            pos_x = random.randint(
                MARGIN_NO_OBSTICLE, self.width - MARGIN_NO_OBSTICLE)
            pos_y = random.randint(
                MARGIN_NO_OBSTICLE, self.height - MARGIN_NO_OBSTICLE)
            width = random.randint(10, 40)
            self.sand[pos_x: pos_x + width, pos_y: pos_y + width] = 1
            rectangles.append([pos_x, pos_y, width])
            #
        self.canvas.add(Color(0.8, 0.7, 0))
        for rect in rectangles:
            pos_x = rect[0]
            pos_y = rect[1]
            width = rect[2]
            self.canvas.add(Rectangle(pos=(pos_x, pos_y), size=(width, width)))

    def get_sand(self):
        return self.sand

    def set_size(self, width, height):
        self.width = width
        self.height = height
        self.sand = np.zeros((width, height))
        # self.sand = np.random.randint(0, 1, size=(width, height))
