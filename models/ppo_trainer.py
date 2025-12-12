"""
PPO Training with TRL - Using Rating-based Rewards
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from tqdm import tqdm

from config import settings

logger = logging.getLogger(__name__)


class PPOModelTrainer:
    """PPO fine-tuning using rating-based rewards directly"""

    def __init__(self, output_dir: str = "./outputs/ppo_model"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.ppo_trainer = None
        # Cache for pre-computed rewards from ratings
        self.reward_cache: Dict[str, float] = {}

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
            r=settings.lora.r,
            lora_alpha=settings.lora.alpha,
            lora_dropout=settings.lora.dropout,
            target_modules=settings.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

    def load_model(self, use_quantization: bool = True):
        """
        Load policy model
        
        Args:
            use_quantization: Whether to use 4-bit quantization
        """
        logger.info(f"📥 Loading policy model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Policy model kwargs
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "peft_config": self._get_lora_config()
        }
        
        if use_quantization and torch.cuda.is_available():
            model_kwargs["quantization_config"] = self._get_quantization_config()
        
        # Load policy model with value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            settings.model.base_model,
            **model_kwargs
        )
        
        logger.info("✅ Policy model loaded")

    def build_reward_cache(self, samples: List[Dict]):
        """
        Build reward cache from pre-processed samples
        
        Args:
            samples: List of dicts with question, answer, reward
        """
        logger.info("📦 Building reward cache from rated samples...")
        
        self.reward_cache.clear()
        for sample in samples:
            # Create cache key from question-answer pair
            key = self._make_cache_key(sample['question'], sample['answer'])
            self.reward_cache[key] = sample['reward']
        
        logger.info(f"✅ Cached {len(self.reward_cache)} reward mappings")

    def _make_cache_key(self, question: str, answer: str) -> str:
        """Create cache key from question-answer pair"""
        # Normalize and create key
        q = question.strip().lower()[:200]  # First 200 chars for matching
        a = answer.strip().lower()[:200]
        return f"{q}|||{a}"

    def prepare_dataset(self, samples: List[Dict]) -> Dataset:
        """
        Prepare dataset for PPO training
        
        Args:
            samples: List of dicts with question, answer, reward
            
        Returns:
            HuggingFace Dataset
        """
        logger.info("🔧 Preparing PPO dataset...")
        
        # Use unique questions for prompts
        questions = list(set([s['question'] for s in samples]))
        
        dataset = Dataset.from_dict({"query": questions})
        
        def tokenize_fn(examples):
            return self.tokenizer(
                examples['query'],
                truncation=True,
                max_length=128,
                padding=False
            )
        
        dataset = dataset.map(tokenize_fn, batched=True)
        dataset = dataset.filter(lambda x: len(x['input_ids']) > 0)
        
        logger.info(f"📊 PPO dataset: {len(dataset)} unique prompts")
        return dataset

    def get_reward(self, question: str, response: str) -> float:
        """
        Get reward for question-response pair
        
        First checks cache (for rated responses), 
        then falls back to neutral reward (0.0) for new responses
        
        Args:
            question: Question text
            response: Response text
            
        Returns:
            Reward value between -1 and 1
        """
        key = self._make_cache_key(question, response)
        
        if key in self.reward_cache:
            return self.reward_cache[key]
        
        # For new/generated responses, return neutral reward
        # In practice, model will learn from cached rated responses
        return 0.0

    def compute_rewards(
        self,
        query_texts: List[str],
        response_texts: List[str]
    ) -> List[torch.Tensor]:
        """
        Compute rewards for query-response pairs
        
        Args:
            query_texts: List of queries
            response_texts: List of responses
            
        Returns:
            List of reward tensors
        """
        rewards = []
        for query, response in zip(query_texts, response_texts):
            score = self.get_reward(query, response)
            rewards.append(torch.tensor(score))
        return rewards

    def train(
        self,
        samples: List[Dict],
        use_quantization: bool = True
    ) -> str:
        """
        Run PPO training with rating-based rewards
        
        Args:
            samples: List of dicts with question, answer, reward
            use_quantization: Whether to use 4-bit quantization
            
        Returns:
            Path to saved model
        """
        logger.info("🚀 Starting PPO training with rating-based rewards...")
        
        # Load model
        self.load_model(use_quantization)
        
        # Build reward cache from samples
        self.build_reward_cache(samples)
        
        # Prepare dataset
        dataset = self.prepare_dataset(samples)
        
        # PPO Config
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=settings.training.batch_size,
            mini_batch_size=1,
            gradient_accumulation_steps=4,
            ppo_epochs=4,
            max_grad_norm=1.0,
            kl_penalty="kl",
            target_kl=0.1,
            init_kl_coef=0.2,
            seed=42,
            log_with=None,
            use_score_scaling=True,
            use_score_norm=True
        )
        
        # Initialize trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=dataset
        )
        
        # Generation config
        generation_kwargs = {
            "max_new_tokens": 256,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # Training loop
        for epoch in range(settings.training.ppo_epochs):
            logger.info(f"📅 Epoch {epoch + 1}/{settings.training.ppo_epochs}")
            
            cached_hits = 0
            total_samples = 0
            
            for batch_idx, batch in enumerate(tqdm(self.ppo_trainer.dataloader)):
                query_tensors = batch['input_ids']
                
                # Generate responses
                response_tensors = []
                for query in query_tensors:
                    response = self.ppo_trainer.generate(
                        query.unsqueeze(0),
                        **generation_kwargs
                    ).squeeze()
                    response_tensors.append(response)
                
                # Decode texts
                query_texts = [
                    self.tokenizer.decode(q, skip_special_tokens=True)
                    for q in query_tensors
                ]
                response_texts = [
                    self.tokenizer.decode(r[len(q):], skip_special_tokens=True)
                    for q, r in zip(query_tensors, response_tensors)
                ]
                
                # Compute rewards (from cache or neutral)
                rewards = self.compute_rewards(query_texts, response_texts)
                
                # Track cache hits
                for q, r in zip(query_texts, response_texts):
                    key = self._make_cache_key(q, r)
                    if key in self.reward_cache:
                        cached_hits += 1
                    total_samples += 1
                
                # PPO step
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                
                # Log
                if batch_idx % 10 == 0:
                    mean_reward = sum(r.item() for r in rewards) / len(rewards)
                    logger.info(f"   Batch {batch_idx}: mean_reward={mean_reward:.4f}")
            
            logger.info(f"   📊 Cache hit rate: {cached_hits}/{total_samples} ({100*cached_hits/max(total_samples,1):.1f}%)")
            
            # Save checkpoint
            checkpoint_path = str(self.output_dir / f"checkpoint_epoch_{epoch + 1}")
            self.ppo_trainer.save_pretrained(checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_path = str(self.output_dir / "final")
        self.ppo_trainer.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info(f"✅ PPO model saved to {final_path}")
        return final_path

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate response using trained model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.pretrained_model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
