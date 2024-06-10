import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from pathlib import Path


class LinearQNet(nn.Module):
    """
    LinearQNet class to represent the Q Network
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the LinearQNet class

        :param input_size: - The input size (11)
        :param hidden_size: - The hidden size
        :param output_size: - The output size (3)
        """

        # Initialize the parent class
        super().__init__()

        # Initialize the layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model

        :param x: - The input tensor
        :return: - The output tensor
        """

        # Pass the input tensor through the first linear layer then return the output tensor
        x = f.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, model_number="00"):
        """
        Save the model to the models directory

        :param model_number: - The model number
        """

        # 1. Create models directory
        model_path = Path("../assets/models")
        model_path.mkdir(parents=True, exist_ok=True)

        # 2. Create a model save path
        model_save_path = model_path / f"model{model_number}.pth"

        # 3. Save the model state dict
        print(f"Saving model to: {model_save_path}")
        torch.save(obj=self.state_dict(), f=model_save_path)


class QTrainer:
    """
    QTrainer class to train the model
    """

    def __init__(self, model, lr, gamma):
        """
        Initialize the QTrainer class

        :param model: - The model
        :param lr: - The learning rate
        :param gamma: - The gamma value
        """

        # Initialize the attributes
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        """
        Train the model with the given state, action, reward, next state, and game over

        :param state: - The state
        :param action: - The action
        :param reward: - The reward
        :param next_state: - The next state
        :param game_over: - True if the game is over, False otherwise
        """

        # Convert the data to tensors
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # Check if the state is a single state
        if len(state.shape) == 1:
            # Add a dimension to the state
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        # 1. predicted Q values with current state
        pred = self.model(state)

        # 2. Q_new = r + y * max(next_predicted Q value) -> only do this if not game_over
        target = pred.clone()
        for idx in range(len(game_over)):
            q_new = reward[idx]
            if not game_over[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = q_new

        # 3. Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


def load_if_exist(model_number) -> 'LinearQNet':
    """
    Check if the model{model_number}.pth exists then load it else return the model

    :param model_number: - The model number
    :return: The model
    """

    # 1. Create models directory
    model_path = Path("../assets/models")
    model_path.mkdir(parents=True, exist_ok=True)

    # 2. Create a model save path
    model_save_path = model_path / f"model{model_number}.pth"

    # 3. Load the model if it exists
    if model_save_path.exists():
        print(f"Loading model from: {model_save_path}")
        model = LinearQNet(input_size=11, hidden_size=256, output_size=3)
        model.load_state_dict(torch.load(model_save_path))
        return model
    else:
        print(f"Model not found at: {model_save_path}, creating a new model.")
        return LinearQNet(input_size=11, hidden_size=256, output_size=3)
