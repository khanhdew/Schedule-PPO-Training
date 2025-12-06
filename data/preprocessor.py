"""
Data preprocessing for reward model training
"""
import logging
from typing import List, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess feedback data for training"""

    def clean_feedback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess feedback data
        
        Args:
            df: Raw feedback DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("🔧 Cleaning feedback data...")
        
        # Make a copy
        df = df.copy()
        
        # Convert rating to numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Remove rows with missing question or answer
        df = df.dropna(subset=['question', 'answer'])
        
        # Remove empty strings
        df = df[df['question'].str.strip() != '']
        df = df[df['answer'].str.strip() != '']
        
        # Strip whitespace
        df['question'] = df['question'].str.strip()
        df['answer'] = df['answer'].str.strip()
        
        logger.info(f"📊 After cleaning: {len(df)} records")
        logger.info(f"   👍 Positive (rating=1): {len(df[df['rating'] == 1])}")
        logger.info(f"   👎 Negative (rating=-1): {len(df[df['rating'] == -1])}")
        
        return df

    def create_preference_pairs(self, df: pd.DataFrame) -> List[Dict]:
        """
        Create chosen/rejected pairs for reward model training
        
        Strategy:
        - Pair positive and negative responses with same/similar questions
        - If no direct pairs, use positive as chosen and any negative as rejected
        
        Args:
            df: Cleaned feedback DataFrame
            
        Returns:
            List of preference pair dictionaries
        """
        logger.info("🔧 Creating preference pairs...")
        
        positive = df[df['rating'] == 1].copy()
        negative = df[df['rating'] == -1].copy()
        
        if len(positive) == 0:
            logger.warning("⚠️ No positive feedback found!")
            return []
        
        if len(negative) == 0:
            logger.warning("⚠️ No negative feedback found!")
            return []
        
        pairs = []
        used_questions = set()
        
        # Method 1: Direct pairing by exact question match
        for _, pos_row in positive.iterrows():
            question = pos_row['question']
            matching_neg = negative[negative['question'] == question]
            
            if len(matching_neg) > 0:
                neg_row = matching_neg.iloc[0]
                pairs.append({
                    'prompt': question,
                    'chosen': pos_row['answer'],
                    'rejected': neg_row['answer']
                })
                used_questions.add(question)
        
        # Method 2: Pair remaining positives with random negatives
        for _, pos_row in positive.iterrows():
            question = pos_row['question']
            if question not in used_questions:
                neg_row = negative.sample(1).iloc[0]
                pairs.append({
                    'prompt': question,
                    'chosen': pos_row['answer'],
                    'rejected': neg_row['answer']
                })
        
        logger.info(f"📊 Created {len(pairs)} preference pairs")
        return pairs

    def format_for_reward_training(self, pairs: List[Dict], tokenizer) -> Dict:
        """
        Format preference pairs for TRL RewardTrainer
        
        Args:
            pairs: List of preference pair dictionaries
            tokenizer: HuggingFace tokenizer
            
        Returns:
            Dictionary ready for Dataset.from_dict()
        """
        from config import settings
        
        max_length = settings.model.max_length
        
        chosen_texts = [
            f"Câu hỏi: {p['prompt']}\n\nTrả lời: {p['chosen']}" 
            for p in pairs
        ]
        rejected_texts = [
            f"Câu hỏi: {p['prompt']}\n\nTrả lời: {p['rejected']}" 
            for p in pairs
        ]
        
        chosen_enc = tokenizer(
            chosen_texts,
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        rejected_enc = tokenizer(
            rejected_texts,
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        
        return {
            'input_ids_chosen': chosen_enc['input_ids'],
            'attention_mask_chosen': chosen_enc['attention_mask'],
            'input_ids_rejected': rejected_enc['input_ids'],
            'attention_mask_rejected': rejected_enc['attention_mask']
        }


# Singleton instance
preprocessor = DataPreprocessor()
