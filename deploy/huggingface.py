"""
HuggingFace Hub deployment
"""
import logging
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
from peft import PeftModel

from config import settings

logger = logging.getLogger(__name__)


class HuggingFaceDeployer:
    """Deploy trained models to HuggingFace Hub"""

    def __init__(self):
        self.api = HfApi()

    def login(self):
        """Authenticate with HuggingFace"""
        if not settings.hf.token:
            raise ValueError("HF_TOKEN not set in environment")
        
        login(token=settings.hf.token)
        logger.info("✅ Logged in to HuggingFace")

    def merge_lora_weights(self, lora_model_path: str) -> tuple:
        """
        Merge LoRA weights into base model
        
        Args:
            lora_model_path: Path to LoRA adapter weights
            
        Returns:
            Tuple of (merged_model, tokenizer)
        """
        logger.info("🔀 Merging LoRA weights...")
        
        # Create offload directory for large models
        offload_dir = Path("./offload_temp")
        offload_dir.mkdir(exist_ok=True)
        
        # Load base model with offloading support
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.model.base_model,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=str(offload_dir)
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            lora_model_path,
            trust_remote_code=True
        )
        
        # Load and merge LoRA - don't use device_map in PeftModel
        model = PeftModel.from_pretrained(
            base_model, 
            lora_model_path,
            is_trainable=False
        )
        merged_model = model.merge_and_unload()
        
        logger.info("✅ LoRA weights merged")
        return merged_model, tokenizer

    def deploy(
        self,
        model_path: str,
        repo_id: str = None,
        commit_message: str = None,
        private: bool = False,
        create_tag: bool = True
    ) -> str:
        """
        Deploy model to HuggingFace Hub
        
        Args:
            model_path: Path to trained model (with LoRA weights)
            repo_id: Target repository (default: from settings)
            commit_message: Commit message
            private: Whether repo should be private
            create_tag: Whether to create version tag
            
        Returns:
            URL of deployed model
        """
        self.login()
        
        repo_id = repo_id or settings.hf.model_repo
        version_tag = datetime.now().strftime("%Y%m%d_%H%M")
        commit_message = commit_message or f"PPO fine-tuned model - {version_tag}"
        
        logger.info(f"📤 Deploying to: {repo_id}")
        
        # Merge LoRA weights
        model, tokenizer = self.merge_lora_weights(model_path)
        
        # Push model
        model.push_to_hub(
            repo_id,
            commit_message=commit_message,
            private=private
        )
        
        # Push tokenizer
        tokenizer.push_to_hub(
            repo_id,
            commit_message=f"Tokenizer update - {version_tag}"
        )
        
        logger.info(f"✅ Model pushed to {repo_id}")
        
        # Create tag
        if create_tag:
            self.api.create_tag(
                repo_id=repo_id,
                tag=f"v{version_tag}",
                tag_message=f"PPO training run {version_tag}"
            )
            logger.info(f"✅ Created tag: v{version_tag}")
        
        # Update model card
        self._update_model_card(repo_id, version_tag)
        
        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"🎉 Deployment complete: {url}")
        return url

    def _update_model_card(self, repo_id: str, version_tag: str):
        """Update model card README"""
        model_card = f"""---
language:
- vi
tags:
- history
- vietnamese
- ppo
- rlhf
license: apache-2.0
---

# HistoryGPT

Vietnamese History AI Assistant fine-tuned with RLHF (PPO).

## Training Details

- **Base Model**: {settings.model.base_model}
- **Fine-tuning**: PPO with human feedback from OpenWebUI
- **Last Updated**: {datetime.now().strftime("%Y-%m-%d")}
- **Version**: {version_tag}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

prompt = "Hãy kể về lịch sử Việt Nam"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

## Training Pipeline

This model was trained using an automated RLHF pipeline:
1. Collect user feedback from OpenWebUI
2. Train reward model from preference pairs
3. Fine-tune with PPO using the reward model
4. Deploy to HuggingFace Hub
"""
        
        self.api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        logger.info("✅ Model card updated")


# Singleton instance
deployer = HuggingFaceDeployer()
