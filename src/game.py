import numpy as np
import pygame
import random
from enum import Enum
from collections import namedtuple

# Initialize pygame and font
pygame.init()
font = pygame.font.Font('../arial.ttf', 25)


class Direction(Enum):
    """
    Enum class to represent the direction of the snake
    """

    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Point namedtuple to represent the position of the snake
Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (65, 111, 228)
BLUE2 = (77, 124, 246)
BLUE3 = (28, 70, 158)
GREEN1 = (162, 209, 72)
GREEN2 = (170, 215, 80)

BLOCK_SIZE = 20


class SnakeGameAI:

    def __init__(self, w=840, h=480, speed=20):
        """
        Initialize the SnakeGameAI class

        :param w: - Width of the game display
        :param h: - Height of the game display
        """

        # Init sizes
        self.w = w
        self.h = h
        self.speed = speed

        # Init parameters
        self.direction = None
        self.head = None
        self.snake = None
        self.food = None
        self.score = None
        self.frame_iteration = None

        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Init game state
        self.reset()

    def reset(self):
        """
        Reset the game state
        """

        # Default direction
        self.direction = Direction.RIGHT

        # Default head position
        self.head = Point(self.w / 2, self.h / 2)
        # Default snake position
        self.snake = [
            self.head,  # Head
            Point(self.head.x - BLOCK_SIZE, self.head.y),  # First body
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)  # Second body
        ]

        # Init game state
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """
        Place the food in the game display
        """

        # Random position
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        # Update food position
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action) -> tuple:
        """
        Play a step in the game

        :param action: - Action to take
        :return: reward, game_over, score
        """

        self.frame_iteration += 1

        # 1. Collect user input (quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, point=None) -> bool:
        """
        Check if the snake has collided with the boundary or itself

        :param point: - Point to check
        :return: True if collision, False otherwise
        """

        # Check if point is None
        if point is None:
            point = self.head
        # Hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # Hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """
        Update the game display
        """

        # Set background color (grid with 20x20 blocks GREEN1 and GREEN2)
        self.display.fill(GREEN1)
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                if (x + y) // BLOCK_SIZE % 2 == 0:
                    pygame.draw.rect(self.display, GREEN2, [x, y, BLOCK_SIZE, BLOCK_SIZE])

        # Draw the snake
        for pt in self.snake:
            pygame.draw.rect(
                self.display,
                (BLUE3 if pt == self.head else BLUE1),  # BLUE3 for head, BLUE1 for body
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))  # Snake inner body

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))  # Food

        # Display the score, with a black background
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Move the snake based on the action

        :param action: - Action to take -> [Straight, right, left]
        """

        # [1, 0, 0] -> Straight
        # [0, 1, 0] -> Right
        # [0, 0, 1] -> Left
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [0, 1, 0]):
            index = (index + 1) % 4  # Right turn (r -> d -> l -> u)
        elif np.array_equal(action, [0, 0, 1]):
            index = (index - 1) % 4  # Left turn (r -> u -> l -> d)
        # else no change (straight)

        # Update direction
        self.direction = clock_wise[index]
        x = self.head.x
        y = self.head.y

        # Move the snake
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        # Update head
        self.head = Point(x, y)
