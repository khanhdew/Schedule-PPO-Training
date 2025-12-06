# PPO Training Pipeline

Automated RLHF pipeline for fine-tuning LLMs using feedback from OpenWebUI.

## 🏗️ Project Structure

```
PPO_Training/
├── config/
│   └── settings.py          # Configuration management
├── data/
│   ├── extractor.py          # PostgreSQL data extraction
│   └── preprocessor.py       # Data preprocessing
├── models/
│   ├── reward_trainer.py     # Reward model training
│   └── ppo_trainer.py        # PPO training
├── deploy/
│   └── huggingface.py        # HuggingFace deployment
├── main.py                   # Main entry point
├── scheduler.py              # Scheduled job runner
├── requirements.txt          # Dependencies
└── .env.example              # Environment template
```

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Run pipeline once**
   ```bash
   python main.py
   ```

4. **Run as scheduled service (every 7 days)**
   ```bash
   python scheduler.py
   ```

## ⚙️ Configuration

All settings are in `.env`:

| Variable | Description |
|----------|-------------|
| `DB_HOST` | PostgreSQL host |
| `DB_PORT` | PostgreSQL port (default: 5432) |
| `DB_NAME` | Database name |
| `DB_USER` | Database username |
| `DB_PASSWORD` | Database password |
| `HF_TOKEN` | HuggingFace write token |
| `HF_MODEL_REPO` | Target model repository |
| `BASE_MODEL` | Base model to fine-tune |
| `MIN_FEEDBACK_SAMPLES` | Minimum samples before training |
| `TRAINING_INTERVAL_DAYS` | Days between training runs |

## 🔄 Pipeline Workflow

1. **Extract** - Get feedback from OpenWebUI PostgreSQL
2. **Preprocess** - Create preference pairs (chosen/rejected)
3. **Train Reward Model** - Using TRL RewardTrainer
4. **Train PPO** - Fine-tune with PPOTrainer
5. **Deploy** - Push to HuggingFace Hub
