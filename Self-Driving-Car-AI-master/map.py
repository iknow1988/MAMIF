import numpy as np
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from ai import Dqn
import random
import pandas as pd
import os.path as path
import atexit
from functools import partial

random.seed(9001)
from kivy.core.image import Image

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
speed = 1
gamma = 0.8
# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5, 41, gamma)
print("loading last saved brain...")
brain.load()

# action2rotation = [0,20,-20]
action2rotation = [i for i in range(-20, 21 ,1)]

last_reward = 0

# Initializing the map
first_update = True

# Initializing the last distance
last_distance = 0

experiment = 1
if path.exists('data.csv'):
   data = pd.read_csv('data.csv')
   experiment = data['experiment'].max() + 1
else:
   columns = ['experiment', 'time', 'speed', 'gamma', 'signal1', 'signal2',
              'signal3', 'distance_to_goal', 'action', 'orientation', 'reward']
   data = pd.DataFrame(columns=columns)

sample = []
time = 0


def init():
    global goal_x
    global goal_y
    global first_update
    global goals_y
    goal_x = 10
    goal_y = largeur - 10
    goals_y = [i for i in range(largeur - 10)]
    first_update = False


class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(10, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(10, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(10, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

        # self.canvas.add(Color(1.0, 1.0, 1.0))
        # self.canvas.add(Ellipse(pos=(self.x, self.y), size=(1, 1)))


class Goal(Widget):
    pass


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


class Game(Widget):

    car = ObjectProperty(None)
    goal = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(speed, 0)

    def update(self, obstacles, dt):
        global sand
        global brain
        global last_reward
        global last_distance
        global goal_x
        global goal_y
        global goals_y
        global longueur
        global largeur
        global time

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            obstacles.set_size(longueur + 1, largeur + 1)
            obstacles.load()
            sand = obstacles.get_sand()
        goal_y = min(goals_y, key=lambda x: np.sqrt((self.car.x - goal_x)**2 + (self.car.y - x)**2))
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.goal.pos = (goal_x, goal_y)

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.2, 0).rotate(self.car.angle)
            self.canvas.add(Color(255, 0, 0))
            self.canvas.add(Ellipse(pos=(self.car.x, self.car.y), size=(2, 2)))
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(speed, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 20:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
            last_reward = 2

        last_distance = distance
        sample.append({'experiment': experiment, 'time': time, 'speed': speed, 'gamma': gamma,
                       'signal1': self.car.signal1, 'signal2': self.car.signal2, 'signal3': self.car.signal3,
                       'distance_to_goal': last_distance, 'action': rotation, 'orientation': orientation,
                       'reward': last_reward})
        time = time + 1



class ObstacleWidget(Widget):

    def __init__(self, width, height, **kwargs):
        # make sure we aren't overriding any important functionality
        super(ObstacleWidget, self).__init__(**kwargs)
        self.sand = np.zeros((width, height))
        self.width = width
        self.height = height

    def load(self):
        rectangles = []
        for _ in range(250):
            pos_x = random.randint(40, self.width - 40)
            pos_y = random.randint(40, self.height - 40)
            width = random.randint(10, 40)
            self.sand[pos_x : pos_x + width, pos_y : pos_y + width] = 1
            rectangles.append([pos_x, pos_y, width])
        # self.canvas.add(Color(0.8, 0.7, 0))
        # for rect in rectangles:
        #     pos_x = rect[0]
        #     pos_y = rect[1]
        #     width = rect[2]
        #     self.canvas.add(Rectangle(pos=(pos_x, pos_y), size=(width, width)))



    def get_sand(self):
        return self.sand

    def set_size(self, width, height):
        self.width = width
        self.height = height
        self.sand = np.zeros((width, height))


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        self.obstacles = ObstacleWidget(100, 50)
        Clock.schedule_interval(partial(parent.update, self.obstacles), 1.0/60.0)
        parent.add_widget(self.obstacles)
        return parent


def save_data():
    global data
    print("saving brain...")
    brain.save()
    print("saving data...")
    data = pd.concat([data, pd.DataFrame(sample)])
    data.to_csv('data.csv', index=False)


if __name__ == '__main__':
    atexit.register(save_data)
    CarApp().run()
