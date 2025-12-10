"""
PPO Training Pipeline - Main Entry Point
Using Rating-based Rewards (No Reward Model Training)
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import settings
from data import extractor, preprocessor
from models import PPOModelTrainer
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
    Run the PPO training pipeline with rating-based rewards
    
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
    logger.info(f"📊 Using rating-based rewards (no reward model)")
    logger.info(f"{'='*50}")
    
    try:
        # Step 1: Extract and prepare feedback data
        logger.info("\n📊 Step 1: Extracting feedback data...")
        
        if not extractor.test_connection():
            logger.error("❌ Database connection failed")
            return False
        
        feedback_df = extractor.extract_feedback(days_back=days_back)
        
        if len(feedback_df) == 0:
            logger.warning("⚠️ No feedback data found")
            return False
        
        # Clean and prepare for PPO
        feedback_df = preprocessor.clean_feedback(feedback_df)
        ppo_samples = preprocessor.prepare_for_ppo(feedback_df)
        
        # Check minimum samples
        min_samples = settings.training.min_feedback_samples
        if len(ppo_samples) < min_samples:
            logger.warning(f"⚠️ Not enough samples: {len(ppo_samples)} < {min_samples}")
            logger.info("⏭️ Skipping training. Waiting for more feedback.")
            return False
        
        logger.info(f"✅ Step 1 complete: {len(ppo_samples)} rated samples")
        
        if dry_run:
            logger.info("🏃 Dry run mode - skipping training")
            return True
        
        # Step 2: Train PPO directly with rating-based rewards
        logger.info("\n🚀 Step 2: Training PPO with rating-based rewards...")
        
        ppo_trainer = PPOModelTrainer(output_dir="./outputs/ppo_model")
        ppo_model_path = ppo_trainer.train(ppo_samples)
        
        logger.info(f"✅ Step 2 complete: PPO model at {ppo_model_path}")
        
        # Step 3: Deploy to HuggingFace
        if not skip_deploy:
            logger.info("\n☁️ Step 3: Deploying to HuggingFace...")
            
            url = deployer.deploy(ppo_model_path)
            
            logger.info(f"✅ Step 3 complete: Model at {url}")
        else:
            logger.info("\n⏭️ Step 3: Skipping deployment (--skip-deploy)")
        
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
