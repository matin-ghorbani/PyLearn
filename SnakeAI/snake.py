import arcade


class Snake(arcade.Sprite):
    def __init__(self, game):
        super().__init__()
        self.body = []
        self.width = 32
        self.height = 32

        self.center_x = game.width//2
        self.center_y = game.height//2
        self.change_x = 0
        self.change_y = 0
        
        self.color = arcade.color.GREEN
        self.colorbody = arcade.color.BLUE

        self.speed = 12
        self.score = 0

        self.w = game.width
        self.h = game.height

    def draw(self):
        arcade.draw_rectangle_filled(
            self.center_x, self.center_y, self.width, self.height, self.color)

        for i, part in enumerate(self.body):
            if i % 2 == 0:
                self.colorbody = arcade.color.BLUE
            elif i % 2 == 1:
                self.colorbody = arcade.color.RED

            arcade.draw_rectangle_filled(
                part['x'], part['y'], self.width, self.height, self.colorbody)

    def move(self):
        self.body.append({'x': self.center_x, 'y': self.center_y})
        if len(self.body) > self.score:
            for _ in range(len(self.body)-self.score):
                self.body.pop(0)

        self.center_x += self.change_x * self.speed
        self.center_y += self.change_y * self.speed
        if self.center_x > self.w or self.center_x < 0 or self.center_y > self.h or self.center_y < 0:
            print('outfit')

    def eat(self, fruit):
        del fruit
        self.score += 1
