"""
Reward Model Training with TRL
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model, TaskType

from config import settings

logger = logging.getLogger(__name__)


class RewardModelTrainer:
    """Train reward model from preference pairs"""

    def __init__(self, output_dir: str = "./outputs/reward_model"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """Get 4-bit quantization config"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    def _get_lora_config(self) -> LoraConfig:
        """Get LoRA config from settings"""
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=settings.lora.r,
            lora_alpha=settings.lora.alpha,
            lora_dropout=settings.lora.dropout,
            target_modules=settings.lora.target_modules,
            bias="none"
        )

    def load_model(self, use_quantization: bool = True):
        """
        Load base model and tokenizer
        
        Args:
            use_quantization: Whether to use 4-bit quantization
        """
        logger.info(f"📥 Loading model: {settings.model.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with reward head
        model_kwargs = {
            "num_labels": 1,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        if use_quantization and torch.cuda.is_available():
            model_kwargs["quantization_config"] = self._get_quantization_config()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.model.base_model,
            **model_kwargs
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self._get_lora_config())
        self.model.print_trainable_parameters()
        
        logger.info("✅ Model loaded with LoRA")

    def prepare_dataset(self, preference_pairs: List[Dict]) -> Dataset:
        """
        Prepare dataset for training
        
        Args:
            preference_pairs: List of {prompt, chosen, rejected} dicts
            
        Returns:
            HuggingFace Dataset
        """
        from data.preprocessor import preprocessor
        
        logger.info("🔧 Preparing dataset...")
        
        # Format for reward training
        formatted = preprocessor.format_for_reward_training(
            preference_pairs,
            self.tokenizer
        )
        
        dataset = Dataset.from_dict(formatted)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        logger.info(f"📊 Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
        return dataset

    def train(
        self,
        preference_pairs: List[Dict],
        use_quantization: bool = True
    ) -> str:
        """
        Train reward model
        
        Args:
            preference_pairs: Training data
            use_quantization: Whether to use 4-bit quantization
            
        Returns:
            Path to saved model
        """
        logger.info("🎯 Starting Reward Model training...")
        
        # Load model if not loaded
        if self.model is None:
            self.load_model(use_quantization)
        
        # Prepare data
        dataset = self.prepare_dataset(preference_pairs)
        
        # Training config
        training_args = RewardConfig(
            output_dir=str(self.output_dir / "checkpoints"),
            per_device_train_batch_size=settings.training.batch_size,
            per_device_eval_batch_size=settings.training.batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=settings.training.reward_epochs,
            learning_rate=settings.training.learning_rate,
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            max_length=settings.model.max_length
        )
        
        # Train
        trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        
        # Save
        final_path = str(self.output_dir / "final")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info(f"✅ Reward model saved to {final_path}")
        return final_path

    def score_response(self, question: str, answer: str) -> float:
        """
        Score a single response
        
        Args:
            question: The question/prompt
            answer: The response to score
            
        Returns:
            Reward score
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        text = f"Câu hỏi: {question}\n\nTrả lời: {answer}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.model.max_length,
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits.item()
        
        return score

    def score_batch(self, qa_pairs: List[Dict]) -> List[float]:
        """
        Score multiple responses
        
        Args:
            qa_pairs: List of {question, answer} dicts
            
        Returns:
            List of scores
        """
        scores = []
        for pair in qa_pairs:
            score = self.score_response(pair['question'], pair['answer'])
            scores.append(score)
        return scores

    def load_trained_model(self, model_path: str):
        """
        Load a previously trained reward model
        
        Args:
            model_path: Path to saved model
        """
        from peft import PeftModel
        
        logger.info(f"📥 Loading trained reward model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            settings.model.base_model,
            num_labels=1,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        logger.info("✅ Trained reward model loaded")
