import arcade
from snake import Snake
from apple import Apple
import pandas as pd


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

        self.data = []

    def on_draw(self):
        arcade.start_render()

        self.snake.draw()
        self.food.draw()

        arcade.draw_text(
            f'Score:{self.snake.score}', 350, 60, arcade.color.ALMOND, 20, 10)

        arcade.finish_render()

    def on_update(self, delta_time: float):
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

            # Direction Status
            'direction': None
        }

        distance_x = self.snake.center_x - self.food.center_x
        distance_y = self.snake.center_y - self.food.center_y

        if self.snake.center_y < self.food.center_y:
            self.snake.change_x = 0
            self.snake.change_y = 1
            data['direction'] = UP

        elif self.snake.center_x < self.food.center_x:
            self.snake.change_x = 1
            self.snake.change_y = 0
            data['direction'] = RIGHT

        elif self.snake.center_y > self.food.center_y:
            self.snake.change_x = 0
            self.snake.change_y = -1
            data['direction'] = DOWN

        elif self.snake.center_x > self.food.center_x:
            self.snake.change_x = -1
            self.snake.change_y = 0
            data['direction'] = LEFT

        # TODO: Apple is on the up
        if self.snake.center_y < self.food.center_y and self.snake.center_x == self.food.center_x:
            data['apple_up'] = 1
            data['apple_right'] = 0
            data['apple_down'] = 0
            data['apple_left'] = 0
            print('Apple is on the up')

        # TODO: Apple is on the right
        elif self.snake.center_x < self.food.center_x and self.snake.center_y == self.food.center_y:
            data['apple_up'] = 0
            data['apple_right'] = 1
            data['apple_down'] = 0
            data['apple_left'] = 0
            print('Apple is on the right')

        # TODO: Apple is on the down
        elif self.snake.center_y > self.food.center_y and self.snake.center_x == self.food.center_x:
            data['apple_up'] = 0
            data['apple_right'] = 0
            data['apple_down'] = 1
            data['apple_left'] = 0
            print('Apple is on the down')

        # TODO: Apple is on the left
        elif self.snake.center_x > self.food.center_x and self.snake.center_y == self.food.center_y:
            data['apple_up'] = 0
            data['apple_right'] = 0
            data['apple_down'] = 0
            data['apple_left'] = 1
            print('Apple is on the left')

        # TODO: Wall and apple distances
        data['wall_up'] = game.height - self.snake.center_y
        data['wall_down'] = self.snake.center_y
        data['wall_left'] = self.snake.center_x
        data['wall_right'] = game.width - self.snake.center_x

        data['distance_x'] = distance_x
        data['distance_y'] = distance_y

        # If the apple is on the left
        if distance_x > 0:
            # If the apple is on the down
            if distance_y > 0:
                self.snake.change_x = -1
                self.snake.change_y = -1
                data['direction'] = LEFT
                if self.snake.change_x == 0:
                    data['direction'] = DOWN

            # If the apple is on the up
            elif distance_y < 0:
                self.snake.change_x = -1
                self.snake.change_y = 1
                data['direction'] = LEFT
                if self.snake.change_x == 0:
                    data['direction'] = UP

            # If the apple is exactly on the left
            else:
                self.snake.change_x = -1
                self.snake.change_y = 0
                data['direction'] = LEFT

        # If the apple is on the right
        if distance_x < 0:
            # If the apple is on the down
            if distance_y > 0:
                self.snake.change_x = 1
                self.snake.change_y = -1
                data['direction'] = RIGHT
                if self.snake.change_x == 0:
                    data['direction'] = DOWN

            # If the apple is on the up
            elif distance_y < 0:
                self.snake.change_x = 1
                self.snake.change_y = 1
                data['direction'] = RIGHT
                if self.snake.change_x == 0:
                    data['direction'] = UP

            # If the apple is exactly on the right
            else:
                self.snake.change_x = 1
                self.snake.change_y = 0
                data['direction'] = RIGHT

            # If the apple and the snake is on the same x
            if distance_x == 0:
                # If the apple is on the down
                if distance_y > 0:
                    self.snake.change_x = 0
                    self.snake.change_y = -1
                    data['direction'] = DOWN

                # If the apple is on the up
                elif distance_y < 0:
                    self.snake.change_x = 0
                    self.snake.change_y = 1
                    data['direction'] = UP

                # If the apple and the snake is on the same x and y
                else:
                    self.snake.change_x = 0
                    self.snake.change_y = 0

        apples_condition: bool = data['apple_up'] != None and data[
            'apple_right'] != None and data['apple_down'] != None and data['apple_left'] != None
        distances_condition: bool = data['distance_x'] != None and data['distance_y'] != None
        walls_condition: bool = data['wall_up'] != None and data[
            'wall_right'] != None and data['wall_down'] != None and data['wall_left'] != None

        # TODO: Save the new data if everything is OK
        if apples_condition and distances_condition and walls_condition:
            self.data.append(data)

        self.snake.on_update(delta_time)
        self.food.on_update()

        if arcade.check_for_collision(self.snake, self.food):
            self.snake.eat(self.food)
            self.score += 1
            self.food = Apple(self)

    def on_key_press(self, symbol: int, modifiers: int):
        # TODO: Quit and save the data as a Pandas DataFrame and save it as csv file
        if symbol == arcade.key.ESCAPE:
            df = pd.DataFrame(self.data)
            df.to_csv('dataset/dataset_version2.csv', index=False)

            arcade.close_window()
            arcade.exit()


if __name__ == '__main__':
    game = Game()
    arcade.run()
