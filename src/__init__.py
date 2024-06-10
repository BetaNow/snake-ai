# Importing all the modules
__all__ = ['game', 'agent', 'model', 'helper']

# Importing the modules
from agent import train
import logging

# Setting up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Snake AI Initialized")

# Launch the training
train(40)
