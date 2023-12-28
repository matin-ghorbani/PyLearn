import arcade
from snake import Snake
from apple import Apple
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class Game(arcade.Window):
    def __init__(self):
        super().__init__(width=512, height=512, title='Super Snake')
        arcade.set_background_color(arcade.color.KHAKI)

        self.snake = Snake(self)
        self.food = Apple(self)
        self.score = 0

        # self.model: Sequential = load_model('./Snake_weight_60ep.h5')
        self.model: Sequential = load_model('./Snake_weight_60ep_version2.h5')

    def on_draw(self):
        arcade.start_render()

        self.snake.draw()
        self.food.draw()

        arcade.draw_text(
            f'Score:{self.snake.score}', 350, 60, arcade.color.ALMOND, 20, 10)

        arcade.finish_render()

    def on_update(self, delta_time):
        self.snake.move()
        data = {
            # Wall Status
            'wall_up': None,
            'wall_right': None,
            'wall_down': None,
            'wall_left': None,

            # Apple Status
            'apple_up': None,
            'apple_right': None,
            'apple_down': None,
            'apple_left': None,

            # Distance of apple to snake status
            'distance_x': None,
            'distance_y': None,
        }

        distance_x = self.snake.center_x - self.food.center_x
        distance_y = self.snake.center_y - self.food.center_y

        # TODO: Apple is on the up
        if self.snake.center_y < self.food.center_y:
            data['apple_up'] = 1
            data['apple_right'] = 0
            data['apple_down'] = 0            
            data['apple_left'] = 0
            print('Apple is on the up')

        # TODO: Apple is on the right
        elif self.snake.center_x < self.food.center_x:
            data['apple_up'] = 0
            data['apple_right'] = 1
            data['apple_down'] = 0
            data['apple_left'] = 0
            print('Apple is on the right')

        # TODO: Apple is on the down
        elif self.snake.center_y > self.food.center_y:
            data['apple_up'] = 0
            data['apple_right'] = 0
            data['apple_down'] = 1
            data['apple_left'] = 0
            print('Apple is on the down')

        # TODO: Apple is on the left
        elif self.snake.center_x > self.food.center_x:
            data['apple_up'] = 0
            data['apple_right'] = 0
            data['apple_down'] = 0
            data['apple_left'] = 1
            print('Apple is on the left')

        # TODO: Wall and apple distances
        data['wall_up'] = game.height - self.snake.center_y
        data['wall_right'] = game.width - self.snake.center_x
        data['wall_down'] = self.snake.center_y
        data['wall_left'] = self.snake.center_x


        data['distance_x'] = distance_x
        data['distance_y'] = distance_y


        data = pd.DataFrame(data, index=[1])
        data = data.values
        # TODO: Predict on data
        output = self.model.predict(data)
        prediction = output.argmax()

        # TODO: Give snake direction
        if prediction == UP:
            self.snake.change_x = 0
            self.snake.change_y = 1
        elif prediction == RIGHT:
            self.snake.change_x = 1
            self.snake.change_y = 0
        elif prediction == DOWN:
            self.snake.change_x = 0
            self.snake.change_y = -1
        elif prediction == LEFT:
            self.snake.change_x = -1
            self.snake.change_y = 0

        self.snake.on_update()
        self.food.on_update()

        if arcade.check_for_collision(self.snake, self.food):
            self.snake.eat(self.food)
            self.score += 1
            self.food = Apple(self)

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ESCAPE:
            arcade.close_window()
            arcade.exit()


if __name__ == '__main__':
    game = Game()
    arcade.run()
