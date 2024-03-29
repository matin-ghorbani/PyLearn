import arcade

from snake import Snake
from apple import Apple

from utils.face_identification import FaceIdentification

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class Game(arcade.Window):
    def __init__(self, has_access: bool, message: str = None):
        super().__init__(width=800, height=600, title='Super Snake 🐍️')
        arcade.set_background_color(arcade.color.KHAKI)

        self.snake = Snake(self)
        self.food = Apple(self)
        self.score = 0

        self.is_ok = True
        self.has_access = has_access
        self.message = message

    def on_draw(self):
        arcade.start_render()

        if self.has_access:
            if self.is_ok:
                self.snake.draw()
                self.food.draw()
            else:
                self.snake.change_x = 0
                self.snake.change_y = 0
                arcade.draw_text("GAME OVER", self.width // 3,
                                 self.height // 2, arcade.color.RED, 28, 16)

            arcade.draw_text(
                f'Score:{self.snake.score}', 350, 60, arcade.color.ALMOND, 20, 10)
        else:
            arcade.draw_text(
                self.message, self.width // 3, self.height // 2, arcade.color.RED, 20, 10)

        arcade.finish_render()

    def on_update(self, delta_time):
        self.is_ok = self.snake.move()
        self.snake.on_update()
        self.food.on_update()

        if arcade.check_for_collision(self.snake, self.food):
            self.snake.eat(self.food)
            self.score += 1
            self.food = Apple(self)

    def on_key_release(self, key: int, modifiers: int):
        if key == arcade.key.LEFT:
            self.snake.change_x = -1
            self.snake.change_y = 0

        elif key == arcade.key.RIGHT:
            self.snake.change_x = 1
            self.snake.change_y = 0

        elif key == arcade.key.UP:
            self.snake.change_x = 0
            self.snake.change_y = 1

        elif key == arcade.key.DOWN:
            self.snake.change_x = 0
            self.snake.change_y = -1

        elif key == arcade.key.ESCAPE:
            arcade.close_window()
            arcade.exit()


if __name__ == '__main__':
    access, message = FaceIdentification.make_online(
        model='buffalo_s',
        thresh=25,
        face_bank_path='./face_bank.npy',
        camera_id=0,
        is_show=False
    )
    game = Game(access, message)
    arcade.run()
