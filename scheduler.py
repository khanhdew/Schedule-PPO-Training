"""
Scheduled Pipeline Runner

Runs the PPO training pipeline on a configurable schedule.
"""
import logging
import time
from datetime import datetime

import schedule

from config import settings
from main import run_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scheduled_job():
    """Run the pipeline as a scheduled job"""
    logger.info(f"\n{'='*60}")
    logger.info(f"⏰ Scheduled job triggered at {datetime.now()}")
    logger.info(f"{'='*60}\n")
    
    try:
        success = run_pipeline()
        
        if success:
            logger.info("✅ Scheduled job completed successfully")
        else:
            logger.warning("⚠️ Scheduled job completed with warnings")
            
    except Exception as e:
        logger.exception(f"❌ Scheduled job failed: {e}")


def run_scheduler():
    """Start the scheduler"""
    interval_days = settings.training.training_interval_days
    
    logger.info(f"🕐 Starting scheduler...")
    logger.info(f"📅 Pipeline will run every {interval_days} days")
    logger.info(f"📍 Next run: immediately, then every {interval_days} days")
    
    # Schedule the job
    schedule.every(interval_days).days.do(scheduled_job)
    
    # Run once immediately
    logger.info("\n🚀 Running initial pipeline...")
    scheduled_job()
    
    # Keep running
    logger.info("\n⏳ Scheduler running. Press Ctrl+C to stop.")
    
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO Training Scheduler")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help=f"Override training interval (default: {settings.training.training_interval_days} days)"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run once and exit (don't schedule)"
    )
    
    args = parser.parse_args()
    
    if args.interval:
        settings.training.training_interval_days = args.interval
    
    if args.run_once:
        logger.info("🏃 Running once (no scheduling)")
        scheduled_job()
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
