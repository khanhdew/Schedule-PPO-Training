"""
Configuration management for PPO Training Pipeline
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "openwebui")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class HuggingFaceConfig:
    token: str = os.getenv("HF_TOKEN", "")
    model_repo: str = os.getenv("HF_MODEL_REPO", "khanhrill/HistoryGPT")


@dataclass
class ModelConfig:
    base_model: str = os.getenv("BASE_MODEL", "khanhrill/HistoryGPT")
    max_length: int = int(os.getenv("MAX_LENGTH", "512"))


@dataclass
class LoRAConfig:
    r: int = int(os.getenv("LORA_R", "16"))
    alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    dropout: float = float(os.getenv("LORA_DROPOUT", "0.1"))
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class TrainingConfig:
    min_feedback_samples: int = int(os.getenv("MIN_FEEDBACK_SAMPLES", "50"))
    training_interval_days: int = int(os.getenv("TRAINING_INTERVAL_DAYS", "7"))
    reward_epochs: int = int(os.getenv("REWARD_EPOCHS", "3"))
    ppo_epochs: int = int(os.getenv("PPO_EPOCHS", "3"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "4"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))


@dataclass
class Settings:
    db: DatabaseConfig = None
    hf: HuggingFaceConfig = None
    model: ModelConfig = None
    lora: LoRAConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.db is None:
            self.db = DatabaseConfig()
        if self.hf is None:
            self.hf = HuggingFaceConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.training is None:
            self.training = TrainingConfig()


# Global settings instance
settings = Settings()
