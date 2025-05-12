"""
Module for preprocessing social media data.
"""

import re
import pandas as pd
import numpy as np
import nltk
from typing import Dict, List, Optional, Union, Tuple
import logging
import networkx as nx
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Class for preprocessing social media data.
    
    Attributes:
        nltk_resources_downloaded (bool): Flag indicating if NLTK resources have been downloaded
    """
    
    def __init__(self, download_nltk: bool = True):
        """
        Initialize the Preprocessor.
        
        Args:
            download_nltk (bool): Whether to download NLTK resources on initialization
        """
        self.nltk_resources_downloaded = False
        if download_nltk:
            self.download_nltk_resources()
        logger.info("Preprocessor initialized")
    
    def download_nltk_resources(self):
        """Download required NLTK resources."""
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            self.nltk_resources_downloaded = True
            logger.info("NLTK resources downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading NLTK resources: {e}")
    
    def preprocess_profiles(self, profiles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess user profile data.
        
        Args:
            profiles_df (pd.DataFrame): DataFrame containing user profiles
            
        Returns:
            pd.DataFrame: Preprocessed profiles DataFrame
        """
        logger.info(f"Preprocessing {len(profiles_df)} profiles")
        
        # Create a copy to avoid modifying the original
        df = profiles_df.copy()
        
        # Ensure user_id column exists
        if 'user_id' not in df.columns:
            if 'id' in df.columns:
                df['user_id'] = df['id']
            else:
                raise ValueError("DataFrame must have a user_id or id column")
        
        # Normalize text columns
        text_columns = ['name', 'username', 'bio', 'description', 'about']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._normalize_text(x) if pd.notna(x) else x)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Normalize email addresses
        if 'email' in df.columns:
            df['email'] = df['email'].apply(lambda x: x.lower() if isinstance(x, str) else x)
        
        # Standardize location data
        if 'location' in df.columns:
            df['location'] = df['location'].apply(lambda x: self._standardize_location(x) if pd.notna(x) else x)
        
        # Convert date columns to standard format
        date_columns = ['join_date', 'created_at', 'registration_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._standardize_date(x) if pd.notna(x) else x)
        
        logger.info("Profile preprocessing completed")
        return df
    
    def preprocess_posts(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess user posts data.
        
        Args:
            posts_df (pd.DataFrame): DataFrame containing user posts
            
        Returns:
            pd.DataFrame: Preprocessed posts DataFrame
        """
        logger.info(f"Preprocessing {len(posts_df)} posts")
        
        # Create a copy to avoid modifying the original
        df = posts_df.copy()
        
        # Ensure required columns exist
        required_columns = ['user_id', 'content']
        for col in required_columns:
            if col not in df.columns:
                if col == 'user_id' and 'author_id' in df.columns:
                    df['user_id'] = df['author_id']
                elif col == 'content' and 'text' in df.columns:
                    df['content'] = df['text']
                else:
                    logger.warning(f"Column {col} not found in posts DataFrame")
        
        # Clean text content
        if 'content' in df.columns:
            df['content'] = df['content'].apply(lambda x: self._clean_post_content(x) if pd.notna(x) else x)
        
        # Standardize timestamp format
        timestamp_columns = ['timestamp', 'created_at', 'date', 'time']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._standardize_timestamp(x) if pd.notna(x) else x)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Extract hashtags if present
        if 'content' in df.columns:
            df['hashtags'] = df['content'].apply(lambda x: self._extract_hashtags(x) if pd.notna(x) else [])
        
        # Extract mentions if present
        if 'content' in df.columns:
            df['mentions'] = df['content'].apply(lambda x: self._extract_mentions(x) if pd.notna(x) else [])
        
        logger.info("Post preprocessing completed")
        return df
    
    def preprocess_network(self, network: nx.Graph) -> nx.Graph:
        """
        Preprocess network data.
        
        Args:
            network (nx.Graph): NetworkX graph representing user connections
            
        Returns:
            nx.Graph: Preprocessed network graph
        """
        logger.info(f"Preprocessing network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
        
        # Create a copy to avoid modifying the original
        G = network.copy()
        
        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Remove isolated nodes (optional)
        # G.remove_nodes_from(list(nx.isolates(G)))
        
        # Ensure the graph is undirected
        if isinstance(G, nx.DiGraph):
            G = G.to_undirected()
        
        logger.info(f"Network preprocessing completed. Resulting network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing special characters and extra whitespace.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _clean_post_content(self, text: str) -> str:
        """
        Clean post content while preserving hashtags and mentions.
        
        Args:
            text (str): Post content to clean
            
        Returns:
            str: Cleaned post content
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text.
        
        Args:
            text (str): Text to extract hashtags from
            
        Returns:
            List[str]: List of extracted hashtags
        """
        if not isinstance(text, str):
            return []
        
        hashtags = re.findall(r'#(\w+)', text)
        return hashtags
    
    def _extract_mentions(self, text: str) -> List[str]:
        """
        Extract mentions from text.
        
        Args:
            text (str): Text to extract mentions from
            
        Returns:
            List[str]: List of extracted mentions
        """
        if not isinstance(text, str):
            return []
        
        mentions = re.findall(r'@(\w+)', text)
        return mentions
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with missing values
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Fill missing text values with empty string
        text_columns = ['name', 'username', 'bio', 'description', 'about', 'content', 'text', 'location']
        for col in text_columns:
            if col in result.columns:
                result[col] = result[col].fillna('')
        
        # Fill missing numeric values with 0
        numeric_columns = ['followers_count', 'following_count', 'likes', 'comments', 'shares']
        for col in numeric_columns:
            if col in result.columns:
                result[col] = result[col].fillna(0)
        
        return result
    
    def _standardize_location(self, location: str) -> str:
        """
        Standardize location string.
        
        Args:
            location (str): Location string to standardize
            
        Returns:
            str: Standardized location
        """
        if not isinstance(location, str):
            return ""
        
        # Convert to lowercase
        location = location.lower()
        
        # Remove common prefixes/suffixes
        location = re.sub(r'^(located in|from|living in)\s+', '', location)
        
        # Remove special characters
        location = re.sub(r'[^\w\s,]', '', location)
        
        # Remove extra whitespace
        location = re.sub(r'\s+', ' ', location).strip()
        
        return location
    
    def _standardize_date(self, date_str: str) -> str:
        """
        Standardize date string to YYYY-MM-DD format.
        
        Args:
            date_str (str): Date string to standardize
            
        Returns:
            str: Standardized date string
        """
        if not isinstance(date_str, str):
            if isinstance(date_str, (datetime, pd.Timestamp)):
                return date_str.strftime('%Y-%m-%d')
            return ""
        
        # Try different date formats
        date_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%b %d, %Y',
            '%B %d, %Y',
            '%d %b %Y',
            '%d %B %Y'
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If all formats fail, return the original string
        return date_str
    
    def _standardize_timestamp(self, timestamp_str: str) -> str:
        """
        Standardize timestamp string to YYYY-MM-DD HH:MM:SS format.
        
        Args:
            timestamp_str (str): Timestamp string to standardize
            
        Returns:
            str: Standardized timestamp string
        """
        if not isinstance(timestamp_str, str):
            if isinstance(timestamp_str, (datetime, pd.Timestamp)):
                return timestamp_str.strftime('%Y-%m-%d %H:%M:%S')
            return ""
        
        # Try different timestamp formats
        timestamp_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%a %b %d %H:%M:%S %z %Y'  # Twitter format
        ]
        
        for fmt in timestamp_formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
        
        # If all formats fail, try to parse as date only
        return self._standardize_date(timestamp_str) + " 00:00:00"
