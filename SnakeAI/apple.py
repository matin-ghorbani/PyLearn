import arcade
from random import randint


class Fruit(arcade.Sprite):
    def __init__(self, game, path):
        super().__init__(path)
        self.width = 32
        self.height = 32

        self.center_x = self.generate_random_position(1, game.width)
        self.center_y = self.generate_random_position(1, game.height)

        self.change_x = 0
        self.change_y = 0
    
    def generate_random_position(self, min_number: int, max_number: int) -> int:
        return (randint(min_number, max_number) // 8) * 8

class Apple(Fruit):
    def __init__(self, game):
        super().__init__(game, './Resources/apple.png')
