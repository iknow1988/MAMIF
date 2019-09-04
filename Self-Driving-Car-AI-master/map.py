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

random.seed(9001)
from kivy.core.image import Image

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
speed = 3
gamma = 0.8
# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5, 41, gamma)
#action2rotation = [0,20,-20]
action2rotation = [i for i in range(-20, 21 ,1)]

last_reward = 0
scores = []
bird_actions = []

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

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global goals_y
    sand = np.zeros((longueur,largeur))
    # sand = np.load('sand.npy')
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
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

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
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    goal = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(speed, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global bird_actions
        global last_distance
        global goal_x
        global goal_y
        global goals_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
        goal_y = min(goals_y, key=lambda x: np.sqrt((self.car.x - goal_x)**2 + (self.car.y - x)**2))
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        bird_actions.append(rotation)
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        self.goal.pos = (goal_x, goal_y)

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
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

        if distance < 30:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
            last_reward = 1
        last_distance = distance
        sample.append([speed, self.car.signal1, self.car.signal2, self.car.signal3, last_distance, rotation, orientation, last_reward])
        # print("Goal", goal_x, goal_y, rotation)


class MyPaintWidget(Widget):
    pass
    # def on_touch_down(self, touch):
    #     global length, n_points, last_x, last_y
    #     with self.canvas:
    #         Color(0.8,0.7,0)
    #         d = 10.
    #         touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
    #         last_x = int(touch.x)
    #         last_y = int(touch.y)
    #         n_points = 0
    #         length = 0
    #         sand[int(touch.x),int(touch.y)] = 1

    # def on_touch_move(self, touch):
    #     global length, n_points, last_x, last_y
    #     if touch.button == 'left':
    #         touch.ud['line'].points += [touch.x, touch.y]
    #         x = int(touch.x)
    #         y = int(touch.y)
    #         length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
    #         n_points += 1.
    #         density = n_points/(length)
    #         touch.ud['line'].width = int(20 * density + 1)
    #         sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
    #         last_x = x
    #         last_y = y


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        loadbtn = Button(text='Load', size=(40,40))
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(loadbtn)
        return parent

    def load(self, obj):
        global sand
        self.painter.canvas.add(Color(0.8,0.7,0))
        for _ in range(100):
            pos_x = random.randint(1, longueur)
            pos_y = random.randint(1, largeur)
            width = random.randint(10, 30)
            elipse = Ellipse(pos=(pos_x, pos_y), size=(width, width))
            sand[int(pos_x) - width: int(pos_x) + width, int(pos_y) - width: int(pos_y) + width] = 1
            self.painter.canvas.add(elipse)

        print("loading last saved brain...")
        brain.load()


def save_data():
    # global data
    # for i, row in enumerate(sample):
    #     df = {'experiment': experiment, 'time': i, 'speed': row[0], 'gamma': gamma, 'signal1': row[1], 'signal2': row[2],
    #           'signal3': row[3], 'distance_to_goal': row[4], 'action': row[5], 'orientation': row[6], 'reward': row[7]}
    #     data = data.append(df, ignore_index=True)
    #     if i % 1000 == 0:
    #         print("left to save =>", len(sample) - i)
    # data.to_csv('data.csv', index = False)

    print("saving brain...")
    brain.save()

if __name__ == '__main__':
    atexit.register(save_data)
    CarApp().run()
