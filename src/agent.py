import numpy as np
import torch
import random
import json
from collections import deque
from pathlib import Path
from game import SnakeGameAI, Direction, Point
from model import QTrainer, load_if_exist
from helper import plot

MODEL_NUMBER = "00"
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = .01


class Agent:
    """
    Agent class to represent the snake AI
    """

    def __init__(self):
        """
        Initialize the Agent class
        """

        self.n_games = 0
        self.epsilon = 0  # Control randomness
        self.gamma = .9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = load_if_exist(MODEL_NUMBER)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def remember(self, state, action, reward, next_state, game_over):
        """
        Remember the state, action, reward, next state, and game over

        :param state: - The game state
        :param action: - The action taken
        :param reward: - The reward received
        :param next_state: - The next state
        :param game_over: - True if the game is over, False otherwise
        """

        # popleft() if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        """
        Train the long memory of the agent
        """

        # Get a mini sample from the memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        # Unzip the mini sample
        state, action, reward, next_state, game_over = zip(*mini_sample)
        # Train the model
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        """
        Train the short memory of the agent

        :param state: - The game state
        :param action: - The action taken
        :param reward: - The reward received
        :param next_state: - The next state
        :param game_over: - True if the game is over, False otherwise
        """

        # Train the model
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state) -> list:
        """
        Get the action to take

        :param state: - The game state
        :return: The action to take
        """

        # Random move: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # Choose random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Get move from the model
            init_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(init_state)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def get_state(game) -> np.array:
    """
    Get the state of the game

    :param game: - The game object
    :return: The state of the game
    """

    # Save the head, points, and directions
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
        """
        Check for danger

        :param direction_keys: - The direction keys
        :return: True if there is danger, False otherwise
        """
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


def train(speed=20):
    """
    Train the agent
    """

    # Define the variables
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(speed=speed)

    # If there is already data, load it
    data_path = Path("../assets/data")
    data_path.mkdir(parents=True, exist_ok=True)

    if (data_path / f"data_model{MODEL_NUMBER}.json").exists():
        with open(data_path / f"data_model{MODEL_NUMBER}.json", "r") as file:
            data = json.load(file)
            plot_scores = data["scores"]
            plot_mean_scores = data["mean_scores"]
            record = data["record"]
            total_score = data["total_score"]
            agent.n_games = data["n_games"]

    # Train the agent
    while True:
        # Get the old state
        state_old = get_state(game)
        # Get the move
        final_move = agent.get_action(state_old)
        # Perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = get_state(game)
        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)
        # Remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # Train the long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Update the record
            if score > record:
                record = score

            # Print the results
            print(f"Game: {agent.n_games} | Score: {score} | Record: {record}")

            # Generate the plot data
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # Plot the results
            plot(plot_scores, plot_mean_scores, MODEL_NUMBER)

            # Save the model
            agent.model.save()

            # Save the data
            data = {
                "scores": plot_scores,
                "mean_scores": plot_mean_scores,
                "record": record,
                "total_score": total_score,
                "n_games": agent.n_games
            }

            # Save the data
            with open(data_path / f"data_model{MODEL_NUMBER}.json", "w") as file:
                json.dump(data, file)
