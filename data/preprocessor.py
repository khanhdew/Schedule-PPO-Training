"""
Data preprocessing for PPO training with star ratings
"""
import logging
from typing import List, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess feedback data for PPO training with star ratings"""

    def __init__(self, rating_min: float = 1.0, rating_max: float = 5.0):
        """
        Initialize preprocessor
        
        Args:
            rating_min: Minimum rating value (default 1.0)
            rating_max: Maximum rating value (default 5.0)
        """
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.rating_mid = (rating_min + rating_max) / 2

    def clean_feedback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess feedback data
        
        Args:
            df: Raw feedback DataFrame
            
        Returns:
            Cleaned DataFrame with valid ratings
        """
        logger.info("🔧 Cleaning feedback data...")
        logger.info(f"   📋 Input shape: {df.shape}")
        logger.info(f"   📋 Columns: {list(df.columns)}")
        
        # Debug: Show sample raw data
        if len(df) > 0:
            logger.info(f"   🔍 Sample raw data (first row):")
            for col in df.columns:
                val = df.iloc[0][col]
                logger.info(f"      {col}: {repr(val)[:100]}")
        
        # Make a copy
        df = df.copy()
        
        # Convert rating to numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Remove rows with missing rating, question or answer
        df = df.dropna(subset=['rating', 'question', 'answer'])
        
        # Remove empty strings
        df = df[df['question'].str.strip() != '']
        df = df[df['answer'].str.strip() != '']
        
        # Strip whitespace
        df['question'] = df['question'].str.strip()
        df['answer'] = df['answer'].str.strip()
        
        # Log rating distribution
        logger.info(f"📊 After cleaning: {len(df)} records")
        logger.info(f"📈 Rating distribution:")
        for rating in sorted(df['rating'].unique()):
            count = len(df[df['rating'] == rating])
            logger.info(f"   ⭐ Rating {rating}: {count}")
        
        avg_rating = df['rating'].mean()
        logger.info(f"   📊 Average rating: {avg_rating:.2f}")
        
        return df

    def normalize_rating(self, rating: float) -> float:
        """
        Normalize rating to reward scale [-1, 1]
        
        Rating at midpoint (e.g., 3 for 1-5 scale) = 0
        Max rating = 1, Min rating = -1
        
        Args:
            rating: Original star rating
            
        Returns:
            Normalized reward value between -1 and 1
        """
        # Center around midpoint, then scale to [-1, 1]
        normalized = (rating - self.rating_mid) / (self.rating_max - self.rating_mid)
        return max(-1.0, min(1.0, normalized))

    def prepare_for_ppo(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare feedback data for PPO training
        
        Args:
            df: Cleaned feedback DataFrame with ratings
            
        Returns:
            List of dictionaries with question, answer, and normalized reward
        """
        logger.info("🔧 Preparing data for PPO training...")
        
        samples = []
        for _, row in df.iterrows():
            reward = self.normalize_rating(row['rating'])
            samples.append({
                'question': row['question'],
                'answer': row['answer'],
                'rating': row['rating'],
                'reward': reward
            })
        
        # Log reward distribution
        positive = [s for s in samples if s['reward'] > 0]
        neutral = [s for s in samples if s['reward'] == 0]
        negative = [s for s in samples if s['reward'] < 0]
        
        logger.info(f"📊 PPO samples: {len(samples)} total")
        logger.info(f"   ✅ Positive rewards: {len(positive)}")
        logger.info(f"   ⚪ Neutral rewards: {len(neutral)}")
        logger.info(f"   ❌ Negative rewards: {len(negative)}")
        
        return samples

    def get_unique_questions(self, df: pd.DataFrame) -> List[str]:
        """
        Get unique questions for PPO training prompts
        
        Args:
            df: Feedback DataFrame
            
        Returns:
            List of unique question strings
        """
        questions = df['question'].unique().tolist()
        logger.info(f"📊 Found {len(questions)} unique questions")
        return questions


# Singleton instance
preprocessor = DataPreprocessor()
