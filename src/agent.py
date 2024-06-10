import numpy as np
import torch
import random
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = .01


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Control randomness
        self.gamma = .9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        points = {
            "L": Point(head.x - 20, head.y),
            "R": Point(head.x + 20, head.y),
            "U": Point(head.x, head.y - 20),
            "D": Point(head.x, head.y + 20)
        }

        directions = {
            "L": game.direction == Direction.LEFT,
            "R": game.direction == Direction.RIGHT,
            "U": game.direction == Direction.UP,
            "D": game.direction == Direction.DOWN
        }

        def danger_check(direction_keys):
            return any(directions[key] and game.is_collision(points[key]) for key in direction_keys)

        state = [
            # Danger straight
            danger_check(["L", "R", "U", "D"]),

            # Danger right
            danger_check(["U", "D", "R", "L"]),

            # Danger left
            danger_check(["D", "U", "L", "R"]),

            # Move direction
            directions["L"],
            directions["R"],
            directions["U"],
            directions["D"],

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x < game.head.x,  # food right
            game.food.y > game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))  # popleft is MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        state, action, reward, next_state, game_over = zip(*mini_sample)
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # Random move: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get the old state
        state_old = agent.get_state(game)

        # Get the move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # Train the long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game: {agent.n_games} | Score: {score} | Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
