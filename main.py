"""
PPO Training Pipeline - Main Entry Point
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import settings
from data import extractor, preprocessor
from models import RewardModelTrainer, PPOModelTrainer
from deploy import deployer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Create directories
Path("logs").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


def run_pipeline(
    skip_deploy: bool = False,
    days_back: int = None,
    dry_run: bool = False
) -> bool:
    """
    Run the full PPO training pipeline
    
    Args:
        skip_deploy: Skip HuggingFace deployment
        days_back: Only use feedback from last N days
        dry_run: Don't actually train, just validate data
        
    Returns:
        True if successful, False otherwise
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"{'='*50}")
    logger.info(f"🚀 PPO Training Pipeline - Run ID: {run_id}")
    logger.info(f"{'='*50}")
    
    try:
        # Step 1: Extract feedback data
        logger.info("\n📊 Step 1: Extracting feedback data...")
        
        if not extractor.test_connection():
            logger.error("❌ Database connection failed")
            return False
        
        feedback_df = extractor.extract_feedback(days_back=days_back)
        
        if len(feedback_df) == 0:
            logger.warning("⚠️ No feedback data found")
            return False
        
        # Preprocess
        feedback_df = preprocessor.clean_feedback(feedback_df)
        preference_pairs = preprocessor.create_preference_pairs(feedback_df)
        
        # Check minimum samples
        min_samples = settings.training.min_feedback_samples
        if len(preference_pairs) < min_samples:
            logger.warning(f"⚠️ Not enough samples: {len(preference_pairs)} < {min_samples}")
            logger.info("⏭️ Skipping training. Waiting for more feedback.")
            return False
        
        logger.info(f"✅ Step 1 complete: {len(preference_pairs)} preference pairs")
        
        if dry_run:
            logger.info("🏃 Dry run mode - skipping training")
            return True
        
        # Step 2: Train Reward Model
        logger.info("\n🎯 Step 2: Training Reward Model...")
        
        reward_trainer = RewardModelTrainer(output_dir="./outputs/reward_model")
        reward_model_path = reward_trainer.train(preference_pairs)
        
        logger.info(f"✅ Step 2 complete: Reward model at {reward_model_path}")
        
        # Step 2.5: Score unrated responses (optional)
        logger.info("\n📝 Scoring unrated responses...")
        
        unrated_df = extractor.extract_unrated(limit=500)
        if len(unrated_df) > 0:
            reward_trainer.load_trained_model(reward_model_path)
            
            scored = []
            for _, row in unrated_df.iterrows():
                if row.get('question') and row.get('answer'):
                    score = reward_trainer.score_response(row['question'], row['answer'])
                    scored.append({
                        'chat_id': row['chat_id'],
                        'message_id': row['message_id'],
                        'score': score
                    })
            
            logger.info(f"📊 Scored {len(scored)} unrated responses")
        
        # Step 3: Train PPO
        logger.info("\n🚀 Step 3: Training PPO...")
        
        # Get unique questions for PPO training
        questions = list(set([
            row['question'] for row in feedback_df.to_dict('records')
            if row.get('question')
        ]))
        
        ppo_trainer = PPOModelTrainer(output_dir="./outputs/ppo_model")
        ppo_model_path = ppo_trainer.train(questions, reward_model_path)
        
        logger.info(f"✅ Step 3 complete: PPO model at {ppo_model_path}")
        
        # Step 4: Deploy to HuggingFace
        if not skip_deploy:
            logger.info("\n☁️ Step 4: Deploying to HuggingFace...")
            
            url = deployer.deploy(ppo_model_path)
            
            logger.info(f"✅ Step 4 complete: Model at {url}")
        else:
            logger.info("\n⏭️ Step 4: Skipping deployment (--skip-deploy)")
        
        # Done
        logger.info(f"\n{'='*50}")
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"{'='*50}")
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
        return False


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO Training Pipeline")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip HuggingFace deployment")
    parser.add_argument("--days-back", type=int, default=None, help="Only use feedback from last N days")
    parser.add_argument("--dry-run", action="store_true", help="Validate data without training")
    
    args = parser.parse_args()
    
    success = run_pipeline(
        skip_deploy=args.skip_deploy,
        days_back=args.days_back,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
