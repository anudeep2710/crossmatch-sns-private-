"""
Module for generating embeddings from user activity patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import logging
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pytz
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemporalEmbedder:
    """
    Class for generating embeddings from user activity patterns.
    
    Attributes:
        embeddings (Dict): Dictionary to store generated embeddings for each platform
        scalers (Dict): Dictionary to store scalers for each platform
    """
    
    def __init__(self, num_time_bins: int = 24, num_day_bins: int = 7, 
                normalize: bool = True, timezone: str = 'UTC'):
        """
        Initialize the TemporalEmbedder.
        
        Args:
            num_time_bins (int): Number of time bins for hour of day (default: 24 for hourly)
            num_day_bins (int): Number of day bins for day of week (default: 7 for daily)
            normalize (bool): Whether to normalize the embeddings
            timezone (str): Default timezone for timestamp conversion
        """
        self.num_time_bins = num_time_bins
        self.num_day_bins = num_day_bins
        self.normalize = normalize
        self.timezone = timezone
        self.embeddings = {}
        self.scalers = {}
        
        logger.info(f"TemporalEmbedder initialized with {num_time_bins} time bins and {num_day_bins} day bins")
    
    def fit_transform(self, activity_data: pd.DataFrame, platform_name: str, 
                     timestamp_col: str, user_id_col: str, 
                     timezone_col: Optional[str] = None,
                     save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Generate temporal embeddings from user activity data.
        
        Args:
            activity_data (pd.DataFrame): DataFrame containing user activity data
            platform_name (str): Name of the platform
            timestamp_col (str): Name of the column containing timestamps
            user_id_col (str): Name of the column containing user IDs
            timezone_col (str, optional): Name of the column containing timezone information
            save_path (str, optional): Path to save the embeddings
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping user IDs to embeddings
        """
        logger.info(f"Generating temporal embeddings for {platform_name} with {len(activity_data)} activities")
        
        # Check if required columns exist
        if timestamp_col not in activity_data.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")
        
        if user_id_col not in activity_data.columns:
            raise ValueError(f"User ID column '{user_id_col}' not found in data")
        
        # Filter out rows with invalid timestamps
        activity_data = activity_data[activity_data[timestamp_col].notna()]
        
        if len(activity_data) == 0:
            logger.warning(f"No valid activity data found for {platform_name}")
            return {}
        
        # Extract temporal features for each user
        user_embeddings = {}
        
        # Group by user
        user_groups = activity_data.groupby(user_id_col)
        
        for user_id, group in user_groups:
            # Convert timestamps to datetime objects
            timestamps = []
            for ts in group[timestamp_col]:
                if isinstance(ts, str):
                    try:
                        timestamps.append(pd.to_datetime(ts))
                    except:
                        logger.warning(f"Could not parse timestamp: {ts}")
                elif isinstance(ts, (datetime, pd.Timestamp)):
                    timestamps.append(ts)
            
            if not timestamps:
                logger.warning(f"No valid timestamps for user {user_id}")
                continue
            
            # Get timezone for this user if available
            user_timezone = self.timezone
            if timezone_col and timezone_col in group.columns:
                tz_values = group[timezone_col].dropna().unique()
                if len(tz_values) > 0:
                    user_timezone = tz_values[0]
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features(timestamps, user_timezone)
            
            # Store embeddings
            user_embeddings[user_id] = temporal_features
        
        # Normalize embeddings if requested
        if self.normalize:
            # Create a matrix of all embeddings
            users = list(user_embeddings.keys())
            embeddings_matrix = np.vstack([user_embeddings[user] for user in users])
            
            # Normalize
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(embeddings_matrix)
            
            # Store scaler
            self.scalers[platform_name] = scaler
            
            # Update user embeddings
            for i, user in enumerate(users):
                user_embeddings[user] = normalized_embeddings[i]
        
        # Store embeddings
        self.embeddings[platform_name] = user_embeddings
        
        # Save embeddings if path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            # Convert to DataFrame
            embeddings_df = pd.DataFrame.from_dict(user_embeddings, orient='index')
            embeddings_df.index.name = user_id_col
            
            # Save to CSV
            embeddings_df.to_csv(os.path.join(save_path, f"{platform_name}_temporal_embeddings.csv"))
            
            # Save scaler
            if self.normalize:
                joblib.dump(self.scalers[platform_name], os.path.join(save_path, f"{platform_name}_temporal_scaler.joblib"))
        
        logger.info(f"Generated temporal embeddings for {len(user_embeddings)} users")
        return user_embeddings
    
    def transform(self, activity_data: pd.DataFrame, platform_name: str, 
                 timestamp_col: str, user_id_col: str,
                 timezone_col: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Generate temporal embeddings for new activity data.
        
        Args:
            activity_data (pd.DataFrame): DataFrame containing user activity data
            platform_name (str): Name of the platform
            timestamp_col (str): Name of the column containing timestamps
            user_id_col (str): Name of the column containing user IDs
            timezone_col (str, optional): Name of the column containing timezone information
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping user IDs to embeddings
        """
        logger.info(f"Transforming {len(activity_data)} activities for {platform_name}")
        
        # Check if required columns exist
        if timestamp_col not in activity_data.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")
        
        if user_id_col not in activity_data.columns:
            raise ValueError(f"User ID column '{user_id_col}' not found in data")
        
        # Filter out rows with invalid timestamps
        activity_data = activity_data[activity_data[timestamp_col].notna()]
        
        if len(activity_data) == 0:
            logger.warning(f"No valid activity data found for {platform_name}")
            return {}
        
        # Extract temporal features for each user
        user_embeddings = {}
        
        # Group by user
        user_groups = activity_data.groupby(user_id_col)
        
        for user_id, group in user_groups:
            # Convert timestamps to datetime objects
            timestamps = []
            for ts in group[timestamp_col]:
                if isinstance(ts, str):
                    try:
                        timestamps.append(pd.to_datetime(ts))
                    except:
                        logger.warning(f"Could not parse timestamp: {ts}")
                elif isinstance(ts, (datetime, pd.Timestamp)):
                    timestamps.append(ts)
            
            if not timestamps:
                logger.warning(f"No valid timestamps for user {user_id}")
                continue
            
            # Get timezone for this user if available
            user_timezone = self.timezone
            if timezone_col and timezone_col in group.columns:
                tz_values = group[timezone_col].dropna().unique()
                if len(tz_values) > 0:
                    user_timezone = tz_values[0]
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features(timestamps, user_timezone)
            
            # Store embeddings
            user_embeddings[user_id] = temporal_features
        
        # Normalize embeddings if requested and scaler exists
        if self.normalize and platform_name in self.scalers:
            # Create a matrix of all embeddings
            users = list(user_embeddings.keys())
            embeddings_matrix = np.vstack([user_embeddings[user] for user in users])
            
            # Normalize using stored scaler
            normalized_embeddings = self.scalers[platform_name].transform(embeddings_matrix)
            
            # Update user embeddings
            for i, user in enumerate(users):
                user_embeddings[user] = normalized_embeddings[i]
        
        return user_embeddings
    
    def _extract_temporal_features(self, timestamps: List[Union[datetime, pd.Timestamp]], 
                                 timezone: str = 'UTC') -> np.ndarray:
        """
        Extract temporal features from a list of timestamps.
        
        Args:
            timestamps (List[Union[datetime, pd.Timestamp]]): List of timestamps
            timezone (str): Timezone for the timestamps
            
        Returns:
            np.ndarray: Temporal features
        """
        # Initialize feature arrays
        hour_of_day = np.zeros(self.num_time_bins)
        day_of_week = np.zeros(self.num_day_bins)
        
        # Get timezone object
        try:
            tz = pytz.timezone(timezone)
        except:
            logger.warning(f"Invalid timezone: {timezone}. Using UTC.")
            tz = pytz.UTC
        
        # Process each timestamp
        for ts in timestamps:
            # Localize timestamp if it's naive
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=pytz.UTC)
                ts = ts.astimezone(tz)
            
            # Extract hour of day (0-23)
            hour = ts.hour
            hour_bin = int(hour * self.num_time_bins / 24)
            hour_of_day[hour_bin] += 1
            
            # Extract day of week (0-6, Monday=0)
            day = ts.weekday()
            day_bin = int(day * self.num_day_bins / 7)
            day_of_week[day_bin] += 1
        
        # Normalize by total count
        total_count = len(timestamps)
        if total_count > 0:
            hour_of_day = hour_of_day / total_count
            day_of_week = day_of_week / total_count
        
        # Calculate additional features
        
        # Posting frequency (posts per day)
        if len(timestamps) > 1:
            min_date = min(timestamps).date()
            max_date = max(timestamps).date()
            date_range = (max_date - min_date).days + 1
            posting_frequency = len(timestamps) / max(1, date_range)
        else:
            posting_frequency = 0
        
        # Posting regularity (variance in time between posts)
        if len(timestamps) > 1:
            sorted_ts = sorted(timestamps)
            time_diffs = [(sorted_ts[i+1] - sorted_ts[i]).total_seconds() / 3600 
                         for i in range(len(sorted_ts)-1)]
            posting_regularity = np.std(time_diffs) if len(time_diffs) > 0 else 0
        else:
            posting_regularity = 0
        
        # Combine all features
        features = np.concatenate([
            hour_of_day,
            day_of_week,
            [posting_frequency, posting_regularity]
        ])
        
        return features
    
    def get_user_embeddings(self, platform_name: str) -> Dict[str, np.ndarray]:
        """
        Get the stored user embeddings for a platform.
        
        Args:
            platform_name (str): Name of the platform
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping user IDs to embeddings
        """
        if platform_name not in self.embeddings:
            logger.warning(f"No embeddings found for {platform_name}")
            return {}
        
        return self.embeddings[platform_name]
