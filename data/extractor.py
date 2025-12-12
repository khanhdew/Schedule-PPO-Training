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
        print("📊 Extracting feedback data...")  # Use print for notebook
        print(f"   🔗 DB: {self.config.host}:{self.config.port}/{self.config.name}")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # First, let's check how many chats exist
        cursor.execute("SELECT COUNT(*) as cnt FROM chat")
        chat_count = cursor.fetchone()['cnt']
        print(f"   📁 Total chats in database: {chat_count}")
        
        # Check if any messages have annotations
        check_query = """
        SELECT COUNT(*) as cnt
        FROM chat t 
        CROSS JOIN LATERAL json_each(t.chat::json#>'{history, messages}') as message
        WHERE message.value::json->'annotation' IS NOT NULL
        """
        cursor.execute(check_query)
        annotation_count = cursor.fetchone()['cnt']
        print(f"   📝 Messages with annotations: {annotation_count}")
        
        # Also check the structure - let's see a sample annotated message
        sample_query = """
        SELECT 
            message.value::json as msg_json
        FROM chat t 
        CROSS JOIN LATERAL json_each(t.chat::json#>'{history, messages}') as message
        WHERE message.value::json->'annotation' IS NOT NULL
        LIMIT 1
        """
        cursor.execute(sample_query)
        sample = cursor.fetchone()
        if sample:
            print(f"   🔍 Sample annotated message: {str(sample['msg_json'])[:500]}...")
        else:
            print("   ⚠️ No annotated messages found!")
        
        query = FEEDBACK_QUERY
        if days_back:
            cutoff = datetime.now() - timedelta(days=days_back)
            query = query.replace(
                "ORDER BY datetime DESC",
                f"AND to_timestamp((message.value::json->>'timestamp')::bigint) > '{cutoff}'\nORDER BY datetime DESC"
            )
        
        df = pd.read_sql(query, conn)
        cursor.close()
        conn.close()
        
        print(f"📊 Extracted {len(df)} feedback records")
        
        # Debug: Show raw data from pandas
        if len(df) > 0:
            print(f"   📋 Columns: {list(df.columns)}")
            print(f"   🔍 First 3 rows RAW DATA:")
            for idx, row in df.head(3).iterrows():
                print(f"      Row {idx}:")
                print(f"         rating: {repr(row.get('rating', 'N/A'))}")
                print(f"         question: {repr(str(row.get('question', 'N/A'))[:80])}")
                print(f"         answer: {repr(str(row.get('answer', 'N/A'))[:80])}")
        else:
            print("   ⚠️ No feedback records found!")
        
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
