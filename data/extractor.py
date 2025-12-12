"""
Data extraction from OpenWebUI PostgreSQL database
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from config import settings

logger = logging.getLogger(__name__)


FEEDBACK_QUERY = """
SELECT 
    u.email as email,
    message.value::json->>'model' as model,
    to_timestamp((message.value::json->>'timestamp')::bigint) as datetime,
    message.value::json->'annotation'->>'rating' as rating,
    message.value::json->'annotation'->>'reason' as reason,
    message.value::json->'annotation'->>'comment' as comment,
    t.chat::json->'history'->'messages'->(message.value::json->>'parentId')->>'content' as question,
    message.value::json->>'content' AS answer
FROM chat t 
CROSS JOIN LATERAL json_each(t.chat::json#>'{history, messages}') as message
INNER JOIN public.user u ON t.user_id = u.id
WHERE message.value::json->'annotation' IS NOT NULL
ORDER BY datetime DESC;
"""

UNRATED_QUERY = """
SELECT 
    t.id as chat_id,
    message.key as message_id,
    u.email as email,
    message.value::json->>'model' as model,
    to_timestamp((message.value::json->>'timestamp')::bigint) as datetime,
    t.chat::json->'history'->'messages'->(message.value::json->>'parentId')->>'content' as question,
    message.value::json->>'content' AS answer
FROM chat t 
CROSS JOIN LATERAL json_each(t.chat::json#>'{history, messages}') as message
INNER JOIN public.user u ON t.user_id = u.id
WHERE message.value::json->'annotation' IS NULL
  AND message.value::json->>'role' = 'assistant'
ORDER BY datetime DESC
LIMIT 1000;
"""


class DataExtractor:
    """Extract feedback data from OpenWebUI database"""

    def __init__(self):
        self.config = settings.db

    def get_connection(self):
        """Create database connection"""
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.name,
            user=self.config.user,
            password=self.config.password,
            cursor_factory=RealDictCursor
        )

    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            conn = self.get_connection()
            conn.close()
            logger.info("✅ Database connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    def extract_feedback(self, days_back: Optional[int] = None) -> pd.DataFrame:
        """
        Extract feedback from database
        
        Args:
            days_back: Only get feedback from last N days (None = all)
            
        Returns:
            DataFrame with feedback records
        """
        logger.info("📊 Extracting feedback data...")
        logger.info(f"   🔗 DB: {self.config.host}:{self.config.port}/{self.config.name}")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # First, let's check how many chats exist
        cursor.execute("SELECT COUNT(*) as cnt FROM chat")
        chat_count = cursor.fetchone()['cnt']
        logger.info(f"   📁 Total chats in database: {chat_count}")
        
        # Check if any messages have annotations
        check_query = """
        SELECT COUNT(*) as cnt
        FROM chat t 
        CROSS JOIN LATERAL json_each(t.chat::json#>'{history, messages}') as message
        WHERE message.value::json->'annotation' IS NOT NULL
        """
        cursor.execute(check_query)
        annotation_count = cursor.fetchone()['cnt']
        logger.info(f"   📝 Messages with annotations: {annotation_count}")
        
        # Also check the structure - let's see a sample message
        sample_query = """
        SELECT 
            message.value::json as msg_json
        FROM chat t 
        CROSS JOIN LATERAL json_each(t.chat::json#>'{history, messages}') as message
        WHERE message.value::json->>'role' = 'assistant'
        LIMIT 1
        """
        cursor.execute(sample_query)
        sample = cursor.fetchone()
        if sample:
            logger.info(f"   🔍 Sample assistant message structure: {str(sample['msg_json'])[:500]}...")
        else:
            logger.warning("   ⚠️ No assistant messages found in database")
        
        query = FEEDBACK_QUERY
        if days_back:
            cutoff = datetime.now() - timedelta(days=days_back)
            query = query.replace(
                "ORDER BY datetime DESC",
                f"AND to_timestamp((message.value::json->>'timestamp')::bigint) > '{cutoff}'\nORDER BY datetime DESC"
            )
        
        logger.info(f"   📜 Running query:\n{query[:500]}...")
        
        df = pd.read_sql(query, conn)
        cursor.close()
        conn.close()
        
        logger.info(f"📊 Extracted {len(df)} feedback records")
        
        if len(df) > 0:
            logger.info(f"   📋 Columns: {list(df.columns)}")
            logger.info(f"   📋 Sample ratings: {df['rating'].value_counts().to_dict() if 'rating' in df.columns else 'N/A'}")
        else:
            logger.warning("   ⚠️ No feedback records found! Check:")
            logger.warning("      1. Are there any annotations in the database?")
            logger.warning("      2. Is the JSON structure correct?")
            logger.warning("      3. Try running FEEDBACK_QUERY manually in psql")
        
        return df

    def extract_unrated(self, limit: int = 1000) -> pd.DataFrame:
        """
        Extract unrated responses for scoring
        
        Args:
            limit: Maximum number of records
            
        Returns:
            DataFrame with unrated responses
        """
        logger.info("📊 Extracting unrated responses...")
        
        conn = self.get_connection()
        
        query = UNRATED_QUERY.replace("LIMIT 1000", f"LIMIT {limit}")
        df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"📊 Extracted {len(df)} unrated responses")
        return df


# Singleton instance
extractor = DataExtractor()
