"""
Data extraction from OpenWebUI PostgreSQL database
Uses SQLAlchemy for reliable data extraction
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

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
    """Extract feedback data from OpenWebUI database using SQLAlchemy"""

    def __init__(self):
        self.config = settings.db
        self._engine = None

    def get_engine(self):
        """Create SQLAlchemy engine (singleton)"""
        if self._engine is None:
            connection_string = (
                f"postgresql://{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.name}"
            )
            self._engine = create_engine(connection_string)
            print(f"   ✅ Created SQLAlchemy engine")
        return self._engine

    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            print("✅ Database connection successful")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

    def extract_feedback(self, days_back: Optional[int] = None) -> pd.DataFrame:
        """
        Extract feedback from database
        
        Args:
            days_back: Only get feedback from last N days (None = all)
            
        Returns:
            DataFrame with feedback records
        """
        print("📊 Extracting feedback data...")
        print(f"   🔗 DB: {self.config.host}:{self.config.port}/{self.config.name}")
        
        engine = self.get_engine()
        
        with engine.connect() as conn:
            # Check total chats
            result = conn.execute(text("SELECT COUNT(*) as cnt FROM chat"))
            chat_count = result.fetchone()[0]
            print(f"   📁 Total chats in database: {chat_count}")
            
            # Check annotations count
            check_query = """
            SELECT COUNT(*) as cnt
            FROM chat t 
            CROSS JOIN LATERAL json_each(t.chat::json#>'{history, messages}') as message
            WHERE message.value::json->'annotation' IS NOT NULL
            """
            result = conn.execute(text(check_query))
            annotation_count = result.fetchone()[0]
            print(f"   📝 Messages with annotations: {annotation_count}")
            
            # Debug: Get one raw annotated message to see structure
            debug_query = """
            SELECT 
                message.value::json as raw_message
            FROM chat t 
            CROSS JOIN LATERAL json_each(t.chat::json#>'{history, messages}') as message
            WHERE message.value::json->'annotation' IS NOT NULL
            LIMIT 1
            """
            result = conn.execute(text(debug_query))
            row = result.fetchone()
            if row:
                print(f"   🔍 Raw message structure: {str(row[0])[:300]}...")
            
            # Build query
            query = FEEDBACK_QUERY
            if days_back:
                cutoff = datetime.now() - timedelta(days=days_back)
                query = query.replace(
                    "ORDER BY datetime DESC",
                    f"AND to_timestamp((message.value::json->>'timestamp')::bigint) > '{cutoff}'\nORDER BY datetime DESC"
                )
            
            # Execute main query with pandas
            df = pd.read_sql(text(query), conn)
        
        print(f"📊 Extracted {len(df)} feedback records")
        
        # Debug output
        if len(df) > 0:
            print(f"   📋 Columns: {list(df.columns)}")
            print(f"   🔍 First 3 rows:")
            for idx in range(min(3, len(df))):
                row = df.iloc[idx]
                print(f"      Row {idx}:")
                print(f"         rating: {repr(row['rating'])}")
                print(f"         question: {repr(str(row['question'])[:80]) if row['question'] else 'None'}")
                print(f"         answer: {repr(str(row['answer'])[:80]) if row['answer'] else 'None'}")
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
        print("📊 Extracting unrated responses...")
        
        engine = self.get_engine()
        
        query = UNRATED_QUERY.replace("LIMIT 1000", f"LIMIT {limit}")
        
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        print(f"📊 Extracted {len(df)} unrated responses")
        return df


# Singleton instance
extractor = DataExtractor()
