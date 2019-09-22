from functools import partial

from kivy.app import App
from kivy.clock import Clock

from game import Game, ObstacleWidget


class BatApp(App):

    def build(self):
        parent = Game()
        parent.serve_bat()
        self.obstacles = ObstacleWidget(100, 50)
        Clock.schedule_interval(
            partial(parent.update, self.obstacles), 1.0/60.0)
        parent.add_widget(self.obstacles)
        return parent


if __name__ == '__main__':
    # atexit.register(save_data)
    BatApp().run()
