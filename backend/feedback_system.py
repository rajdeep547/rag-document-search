# backend/feedback_system.py
"""
User Feedback Collection & Analytics System
"""
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
import os
from pathlib import Path


class FeedbackSystem:
    """
    Collect, store, and analyze user feedback
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feedback system
        
        Args:
            storage_path: Path to store feedback data
        """
        if storage_path is None:
            storage_path = os.getenv('FEEDBACK_STORAGE', 'storage/feedback.json')
        
        self.storage_path = storage_path
        self.feedback_data = self._load_feedback()
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists"""
        dir_path = Path(self.storage_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_feedback(self) -> List[Dict]:
        """Load feedback from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load feedback: {e}")
                return []
        return []
    
    def _save_feedback(self):
        """Save feedback to storage"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save feedback: {e}")
    
    def add_feedback(self, 
                     question: str, 
                     answer: str, 
                     source: str,
                     rating: int,
                     user_id: Optional[str] = None,
                     comments: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> Dict:
        """
        Add user feedback
        
        Args:
            question: User's question
            answer: System's answer
            source: Source of the answer
            rating: Rating (1-5)
            user_id: Optional user identifier
            comments: Optional comments
            metadata: Additional metadata
            
        Returns:
            Feedback entry dictionary
        """
        # Validate rating
        if not 1 <= rating <= 5:
            rating = 3  # Default to neutral
        
        feedback_entry = {
            'id': len(self.feedback_data) + 1,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'question': question[:500],  # Truncate
            'answer': answer[:1000],  # Truncate
            'source': source[:200],
            'rating': rating,
            'user_id': user_id or 'anonymous',
            'comments': comments or '',
            'metadata': metadata or {}
        }
        
        self.feedback_data.append(feedback_entry)
        self._save_feedback()
        
        return feedback_entry
    
    def get_stats(self) -> Dict:
        """
        Get feedback statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.feedback_data:
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'rating_distribution': {},
                'top_questions': [],
                'recent_feedback': []
            }
        
        df = pd.DataFrame(self.feedback_data)
        
        # Rating distribution
        rating_dist = df['rating'].value_counts().to_dict()
        
        # Top questions
        top_questions = df['question'].value_counts().head(5).to_dict()
        
        # Recent feedback (last 10)
        recent = df.sort_values('timestamp', ascending=False).head(10).to_dict('records')
        
        return {
            'total_feedback': len(self.feedback_data),
            'average_rating': round(df['rating'].mean(), 2),
            'rating_distribution': rating_dist,
            'top_questions': top_questions,
            'recent_feedback': recent,
            'high_rated': len(df[df['rating'] >= 4]),
            'low_rated': len(df[df['rating'] <= 2]),
            'unique_questions': df['question'].nunique()
        }
    
    def get_best_answers(self, min_rating: int = 4, limit: int = 10) -> List[Dict]:
        """
        Get highest rated answers
        
        Args:
            min_rating: Minimum rating threshold
            limit: Maximum number of results
            
        Returns:
            List of best answers
        """
        if not self.feedback_data:
            return []
        
        df = pd.DataFrame(self.feedback_data)
        best = df[df['rating'] >= min_rating].sort_values('rating', ascending=False)
        
        return best.head(limit).to_dict('records')
    
    def get_improvement_suggestions(self) -> List[str]:
        """
        Generate improvement suggestions from feedback
        
        Returns:
            List of suggestions
        """
        suggestions = []
        
        if not self.feedback_data:
            return ["Collect more feedback for insights"]
        
        df = pd.DataFrame(self.feedback_data)
        
        # Low rated answers
        low_rated = df[df['rating'] <= 2]
        if len(low_rated) > 0:
            suggestions.append(f"💡 {len(low_rated)} answers rated low - review these questions")
            # Show sample low-rated question
            sample = low_rated.iloc[0]['question'][:50] + "..." if len(low_rated) > 0 else ""
            if sample:
                suggestions.append(f"   Example: '{sample}'")
        
        # Most common questions
        common_q = df['question'].value_counts().head(3)
        for q, count in common_q.items():
            suggestions.append(f"📌 '{q[:50]}...' asked {count} times - consider adding to FAQ")
        
        # High engagement
        if len(df) > 10:
            suggestions.append(f"📊 Total feedback: {len(df)} - keep collecting for better insights")
        
        # Average rating
        avg = df['rating'].mean()
        if avg < 3:
            suggestions.append(f"⭐ Average rating {avg:.1f}/5 - consider improving answer quality")
        elif avg >= 4:
            suggestions.append(f"⭐ Great average rating {avg:.1f}/5 - keep up the good work!")
        
        return suggestions
    
    def get_feedback_by_question(self, question: str) -> List[Dict]:
        """
        Get feedback for a specific question
        
        Args:
            question: Question string
            
        Returns:
            List of feedback entries
        """
        return [f for f in self.feedback_data if question in f['question']]
    
    def get_feedback_by_rating(self, rating: int) -> List[Dict]:
        """
        Get feedback with specific rating
        
        Args:
            rating: Rating value (1-5)
            
        Returns:
            List of feedback entries
        """
        return [f for f in self.feedback_data if f['rating'] == rating]
    
    def get_feedback_by_date(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Get feedback within date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of feedback entries
        """
        results = []
        for f in self.feedback_data:
            if start_date <= f.get('date', '') <= end_date:
                results.append(f)
        return results
    
    def clear_feedback(self):
        """Clear all feedback"""
        self.feedback_data = []
        self._save_feedback()
        print("🗑️ All feedback cleared")
    
    def export_feedback(self, filename: str = "feedback_export.json") -> str:
        """
        Export feedback to file
        
        Args:
            filename: Output filename
            
        Returns:
            File path
        """
        export_path = f"storage/exports/{filename}"
        
        # Ensure directory exists
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump({
                'exported_at': datetime.now().isoformat(),
                'total_entries': len(self.feedback_data),
                'data': self.feedback_data
            }, f, indent=2, ensure_ascii=False)
        
        return export_path
    
    def get_analytics(self) -> Dict:
        """
        Get comprehensive analytics
        """
        stats = self.get_stats()
        
        # Additional analytics
        if self.feedback_data:
            df = pd.DataFrame(self.feedback_data)
            
            # Daily trend
            daily_trend = df['date'].value_counts().to_dict()
            
            # User engagement
            user_counts = df['user_id'].value_counts().head(10).to_dict()
            
            return {
                **stats,
                'daily_trend': daily_trend,
                'top_users': user_counts
            }
        
        return stats


class EnhancedFeedbackSystem(FeedbackSystem):
    """
    Enhanced feedback system with additional features
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        super().__init__(storage_path)
        self.review_queue = []
    
    def add_feedback_with_review(self, 
                                  question: str,
                                  answer: str,
                                  source: str,
                                  rating: int,
                                  user_id: Optional[str] = None,
                                  comments: Optional[str] = None) -> Dict:
        """Add feedback and queue for review if low rated"""
        
        feedback = self.add_feedback(question, answer, source, rating, user_id, comments)
        
        # Queue for review if rating is low
        if rating <= 2:
            self.review_queue.append({
                'feedback_id': feedback['id'],
                'question': question,
                'rating': rating,
                'timestamp': datetime.now().isoformat()
            })
        
        return feedback
    
    def get_review_queue(self) -> List[Dict]:
        """Get items awaiting review"""
        return self.review_queue
    
    def mark_reviewed(self, feedback_id: int):
        """Mark a feedback item as reviewed"""
        self.review_queue = [q for q in self.review_queue if q['feedback_id'] != feedback_id]