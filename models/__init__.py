"""Models package - PPO training with rating-based rewards"""
from .ppo_trainer import PPOModelTrainer

# RewardModelTrainer is optional - only needed if you want to train a separate reward model
# from .reward_trainer import RewardModelTrainer

__all__ = ['PPOModelTrainer']
