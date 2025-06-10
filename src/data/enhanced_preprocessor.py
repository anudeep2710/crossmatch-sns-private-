"""
Enhanced preprocessor with Named Entity Recognition and data augmentation.
"""

import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import random

try:
    # Try to import spacy for NER
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. NER functionality will be limited.")

class EnhancedPreprocessor:
    """Enhanced preprocessor with NER, data augmentation, and quality filtering."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Download NLTK data if needed (includes initialization of components)
        self._download_nltk_data()
        
        # Initialize spaCy model for NER if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model 'en_core_web_sm' not found. NER functionality disabled.")
        
        # Data quality filters
        self.min_post_length = config.get('min_post_length', 20)
        self.min_posts_per_user = config.get('min_posts_per_user', 5)
        self.max_posts_per_user = config.get('max_posts_per_user', 1000)
        
        # Data augmentation settings
        self.augmentation_prob = config.get('augmentation_prob', 0.3)
        self.augmentation_ratio = config.get('augmentation_ratio', 0.5)
        
        # Location normalization mapping (expanded for regular users)
        self.location_mapping = {
            'nyc': 'new york city',
            'ny': 'new york',
            'sf': 'san francisco',
            'la': 'los angeles',
            'dc': 'washington dc',
            'uk': 'united kingdom',
            'usa': 'united states',
            'us': 'united states',
            'ca': 'california',
            'fl': 'florida',
            'tx': 'texas',
            'chi': 'chicago',
            'philly': 'philadelphia',
            'vegas': 'las vegas',
            'miami': 'miami',
            'boston': 'boston',
            'seattle': 'seattle',
            'denver': 'denver',
            'atlanta': 'atlanta'
        }
        
        # Company normalization mapping (expanded for regular users and common companies)
        self.company_mapping = {
            'google inc': 'google',
            'microsoft corp': 'microsoft',
            'apple inc': 'apple',
            'amazon.com': 'amazon',
            'meta platforms': 'meta',
            'facebook inc': 'meta',
            'ibm corp': 'ibm',
            'oracle corp': 'oracle',
            'salesforce inc': 'salesforce',
            'netflix inc': 'netflix',
            'uber technologies': 'uber',
            'lyft inc': 'lyft',
            'airbnb inc': 'airbnb',
            'tesla inc': 'tesla',
            'spacex': 'spacex',
            'walmart inc': 'walmart',
            'target corp': 'target',
            'starbucks corp': 'starbucks',
            'mcdonalds corp': 'mcdonalds',
            'nike inc': 'nike',
            'adidas ag': 'adidas'
        }
        
        # Common job titles for regular users
        self.job_title_mapping = {
            'software eng': 'software engineer',
            'data scientist': 'data scientist',
            'product mgr': 'product manager',
            'marketing mgr': 'marketing manager',
            'sales rep': 'sales representative',
            'customer service': 'customer service',
            'teacher': 'teacher',
            'nurse': 'nurse',
            'doctor': 'doctor',
            'student': 'student',
            'freelancer': 'freelancer',
            'consultant': 'consultant',
            'entrepreneur': 'entrepreneur'
        }
        
    def _download_nltk_data(self):
        """Download required NLTK data with robust error handling."""
        nltk_downloads = ['punkt_tab', 'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for item in nltk_downloads:
            try:
                # Check if data exists
                if item == 'punkt_tab':
                    try:
                        nltk.data.find('tokenizers/punkt_tab/english/')
                        continue
                    except LookupError:
                        pass
                else:
                    try:
                        if item == 'stopwords':
                            nltk.data.find('corpora/stopwords')
                        elif item == 'wordnet':
                            nltk.data.find('corpora/wordnet')
                        else:
                            nltk.data.find(f'tokenizers/{item}')
                        continue
                    except LookupError:
                        pass
                
                # Download if not found
                try:
                    self.logger.info(f"Downloading NLTK data: {item}")
                    nltk.download(item, quiet=False)
                    self.logger.info(f"Successfully downloaded NLTK data: {item}")
                except Exception as e:
                    self.logger.warning(f"Failed to download NLTK data: {item} - {e}")
                    
            except Exception as e:
                self.logger.warning(f"Error checking/downloading NLTK data {item}: {e}")
        
        # Initialize lemmatizer with error handling
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            self.logger.warning(f"Failed to initialize lemmatizer: {e}")
            self.lemmatizer = None
            
        # Initialize stopwords with error handling
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            self.logger.warning(f"Failed to load stopwords: {e}. Using basic stopwords.")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply enhanced preprocessing to all datasets.
        
        Args:
            data: Dictionary containing platform dataframes
            
        Returns:
            Dictionary with preprocessed dataframes
        """
        processed_data = {}
        
        for platform, df in data.items():
            self.logger.info(f"Preprocessing {platform} data...")
            
            # Apply quality filtering
            df_filtered = self._apply_quality_filters(df, platform)
            
            # Apply text preprocessing
            df_processed = self._preprocess_text_columns(df_filtered)
            
            # Extract and normalize entities
            df_processed = self._extract_and_normalize_entities(df_processed)
            
            # Apply data augmentation if enabled
            if self.config.get('enable_augmentation', False):
                df_processed = self._apply_data_augmentation(df_processed, platform)
            
            processed_data[platform] = df_processed
            
        return processed_data
    
    def _apply_quality_filters(self, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """Apply data quality filters."""
        original_size = len(df)
        
        # Filter by post length if content column exists
        if 'content' in df.columns:
            df = df[df['content'].str.len() >= self.min_post_length]
        
        # Filter users by post count
        if 'user_id' in df.columns:
            user_post_counts = df['user_id'].value_counts()
            valid_users = user_post_counts[
                (user_post_counts >= self.min_posts_per_user) & 
                (user_post_counts <= self.max_posts_per_user)
            ].index
            df = df[df['user_id'].isin(valid_users)]
        
        # Remove duplicate content
        if 'content' in df.columns:
            df = df.drop_duplicates(subset=['content'])
        
        # Filter out likely bot accounts (very simple heuristics)
        df = self._filter_bot_accounts(df)
        
        filtered_size = len(df)
        self.logger.info(f"Quality filtering {platform}: {original_size} -> {filtered_size} "
                        f"({filtered_size/original_size*100:.1f}% retained)")
        
        return df
    
    def _filter_bot_accounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out likely bot accounts using simple heuristics."""
        if 'bio' not in df.columns and 'content' not in df.columns:
            return df
        
        # Check for repetitive content patterns
        content_col = 'content' if 'content' in df.columns else 'bio'
        
        # Remove accounts with very repetitive posting patterns
        if 'user_id' in df.columns and len(df) > 0:
            try:
                user_content_diversity = df.groupby('user_id')[content_col].apply(
                    lambda x: len(set(x.dropna())) / max(1, len(x.dropna())) if len(x.dropna()) > 0 else 0
                )
                # Be more lenient for regular users (lower threshold)
                diverse_users = user_content_diversity[user_content_diversity > 0.2].index
                df = df[df['user_id'].isin(diverse_users)]
            except Exception as e:
                self.logger.warning(f"Bot filtering failed: {e}")
                # Return original data if filtering fails
                pass
        
        return df
    
    def _preprocess_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess text columns."""
        text_columns = ['bio', 'content', 'name', 'location']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._preprocess_text)
                df[f'{col}_processed'] = df[col].apply(self._advanced_text_processing)
        
        return df
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove hashtags and mentions but keep the text
        text = re.sub(r'[#@](\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _advanced_text_processing(self, text: str) -> str:
        """Advanced text processing with lemmatization and robust error handling."""
        if pd.isna(text) or text == '':
            return ''
        
        try:
            # Tokenize with multiple fallback options
            tokens = []
            try:
                tokens = word_tokenize(text)
            except (LookupError, OSError) as e:
                self.logger.warning(f"NLTK tokenization failed: {e}. Using simple tokenization.")
                # Multiple fallback tokenization methods
                try:
                    # Try simple regex tokenization
                    import re
                    tokens = re.findall(r'\b\w+\b', text.lower())
                except:
                    # Ultimate fallback - simple split
                    tokens = text.split()
            
            # Remove stopwords and non-alphabetic tokens
            cleaned_tokens = []
            for token in tokens:
                if token and token.isalpha() and len(token) > 1:
                    if token.lower() not in self.stop_words:
                        cleaned_tokens.append(token.lower())
            
            # Lemmatize with fallback
            try:
                if self.lemmatizer:
                    lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in cleaned_tokens]
                else:
                    lemmatized_tokens = cleaned_tokens
            except Exception as e:
                self.logger.warning(f"Lemmatization failed: {e}. Using original tokens.")
                lemmatized_tokens = cleaned_tokens
            
            return ' '.join(lemmatized_tokens) if lemmatized_tokens else ''
            
        except Exception as e:
            self.logger.warning(f"Advanced text processing failed for text: {text[:50]}... Error: {e}")
            # Final fallback to basic preprocessing
            try:
                return self._preprocess_text(text)
            except:
                # Ultimate fallback - just return cleaned text
                return re.sub(r'[^a-zA-Z\s]', ' ', text.lower()).strip()
    
    def _extract_and_normalize_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and normalize named entities."""
        if self.nlp is None:
            self.logger.warning("spaCy not available. Skipping entity extraction.")
            return df
        
        # Extract entities from bio and content
        for col in ['bio', 'content']:
            if col in df.columns:
                df[f'{col}_entities'] = df[col].apply(self._extract_entities)
                df[f'{col}_normalized'] = df[col].apply(self._normalize_entities)
        
        # Normalize location column specifically
        if 'location' in df.columns:
            df['location_normalized'] = df['location'].apply(self._normalize_location)
        
        return df
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        if pd.isna(text) or text == '' or self.nlp is None:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text.lower())
        
        return entities
    
    def _normalize_entities(self, text: str) -> str:
        """Normalize entities in text."""
        if pd.isna(text) or text == '':
            return text
        
        normalized_text = text.lower()
        
        # Normalize locations
        for abbrev, full in self.location_mapping.items():
            normalized_text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, normalized_text)
        
        # Normalize companies
        for variant, canonical in self.company_mapping.items():
            normalized_text = re.sub(r'\b' + re.escape(variant) + r'\b', canonical, normalized_text)
        
        # Normalize job titles
        for variant, canonical in self.job_title_mapping.items():
            normalized_text = re.sub(r'\b' + re.escape(variant) + r'\b', canonical, normalized_text)
        
        return normalized_text
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location strings."""
        if pd.isna(location) or location == '':
            return ''
        
        location = location.lower().strip()
        
        # Apply location mapping
        for abbrev, full in self.location_mapping.items():
            location = re.sub(r'\b' + abbrev + r'\b', full, location)
        
        # Remove common location suffixes/prefixes
        location = re.sub(r'\b(city|area|region|metro)\b', '', location)
        location = re.sub(r'\s+', ' ', location).strip()
        
        return location
    
    def _apply_data_augmentation(self, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """Apply data augmentation techniques."""
        if len(df) == 0:
            return df
        
        augmented_rows = []
        
        for idx, row in df.iterrows():
            # Randomly decide whether to augment this row
            if random.random() < self.augmentation_ratio:
                augmented_row = self._augment_row(row)
                if augmented_row is not None:
                    augmented_rows.append(augmented_row)
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            df = pd.concat([df, augmented_df], ignore_index=True)
            
            self.logger.info(f"Data augmentation {platform}: added {len(augmented_rows)} rows")
        
        return df
    
    def _augment_row(self, row: pd.Series) -> Optional[pd.Series]:
        """Augment a single row of data."""
        augmented_row = row.copy()
        
        # Augment text content
        if 'content' in row.index and pd.notna(row['content']):
            augmented_content = self._augment_text(row['content'])
            if augmented_content != row['content']:
                augmented_row['content'] = augmented_content
                augmented_row['is_augmented'] = True
                return augmented_row
        
        # Augment bio
        if 'bio' in row.index and pd.notna(row['bio']):
            augmented_bio = self._augment_text(row['bio'])
            if augmented_bio != row['bio']:
                augmented_row['bio'] = augmented_bio
                augmented_row['is_augmented'] = True
                return augmented_row
        
        return None
    
    def _augment_text(self, text: str) -> str:
        """Augment text using simple techniques suitable for regular users."""
        if pd.isna(text) or len(text) < 10:
            return text
        
        # Expanded synonym replacement for everyday language
        synonyms = {
            'good': ['great', 'excellent', 'wonderful', 'amazing', 'awesome'],
            'bad': ['terrible', 'awful', 'poor', 'horrible', 'disappointing'],
            'big': ['large', 'huge', 'massive', 'enormous', 'giant'],
            'small': ['tiny', 'little', 'mini', 'compact', 'petite'],
            'happy': ['joyful', 'pleased', 'delighted', 'excited', 'thrilled'],
            'sad': ['unhappy', 'depressed', 'melancholy', 'disappointed', 'upset'],
            'love': ['adore', 'enjoy', 'appreciate', 'cherish', 'like'],
            'work': ['job', 'career', 'profession', 'employment', 'occupation'],
            'home': ['house', 'place', 'residence', 'dwelling', 'apartment'],
            'friend': ['buddy', 'pal', 'companion', 'mate', 'colleague'],
            'family': ['relatives', 'folks', 'kin', 'loved ones', 'household'],
            'food': ['meal', 'cuisine', 'dish', 'snack', 'dinner'],
            'travel': ['journey', 'trip', 'vacation', 'adventure', 'exploration'],
            'music': ['songs', 'tunes', 'melodies', 'tracks', 'sounds'],
            'movie': ['film', 'cinema', 'show', 'picture', 'flick']
        }
        
        words = text.split()
        augmented_words = []
        
        for word in words:
            if word.lower() in synonyms and random.random() < self.augmentation_prob:
                augmented_words.append(random.choice(synonyms[word.lower()]))
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional features from preprocessed data."""
        if len(df) == 0:
            return df
        
        # Text length features
        if 'content' in df.columns:
            df['content_length'] = df['content'].str.len()
            df['word_count'] = df['content'].str.split().str.len()
            df['avg_word_length'] = df['content'].apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x and len(x.split()) > 0 else 0
            )
        
        if 'bio' in df.columns:
            df['bio_length'] = df['bio'].str.len()
            df['bio_word_count'] = df['bio'].str.split().str.len()
        
        # Engagement features (if available)
        engagement_cols = ['likes', 'shares', 'comments', 'followers']
        for col in engagement_cols:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])
                df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        return df
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return {
            'min_post_length': self.min_post_length,
            'min_posts_per_user': self.min_posts_per_user,
            'augmentation_enabled': self.config.get('enable_augmentation', False),
            'augmentation_ratio': self.augmentation_ratio,
            'spacy_available': self.nlp is not None
        }
    
    def preprocess_profiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess profile data.
        
        Args:
            data: DataFrame containing profile data
            
        Returns:
            Preprocessed profile DataFrame
        """
        processed_data = data.copy()
        
        # Apply text preprocessing to bio and other text fields
        text_fields = ['bio', 'description', 'name', 'location']
        for field in text_fields:
            if field in processed_data.columns:
                processed_data[field] = processed_data[field].fillna('')
                processed_data[field] = processed_data[field].apply(self._preprocess_text)
        
        # Extract additional features
        processed_data = self.extract_features(processed_data)
        
        return processed_data
    
    def preprocess_posts(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess posts/content data.
        
        Args:
            data: DataFrame containing posts data
            
        Returns:
            Preprocessed posts DataFrame
        """
        processed_data = data.copy()
        
        # Apply text preprocessing to content
        if 'content' in processed_data.columns:
            processed_data['content'] = processed_data['content'].fillna('')
            processed_data['content'] = processed_data['content'].apply(self._preprocess_text)
            processed_data['content_processed'] = processed_data['content'].apply(self._advanced_text_processing)
        
        # Extract additional features
        processed_data = self.extract_features(processed_data)
        
        return processed_data
    
    def preprocess_network(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess network/connection data.
        
        Args:
            data: DataFrame containing network data
            
        Returns:
            Preprocessed network DataFrame
        """
        processed_data = data.copy()
        
        # Basic network preprocessing
        if 'source' in processed_data.columns:
            processed_data['source'] = processed_data['source'].astype(str)
        if 'target' in processed_data.columns:
            processed_data['target'] = processed_data['target'].astype(str)
        
        # Remove self-loops
        if 'source' in processed_data.columns and 'target' in processed_data.columns:
            processed_data = processed_data[processed_data['source'] != processed_data['target']]
        
        # Remove duplicates
        processed_data = processed_data.drop_duplicates()
        
        return processed_data