"""
Flexible CSV data loader for various formats and structures.
This module handles loading CSV files with different schemas and automatically
detects the data structure to enable cross-platform user identification.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CSVDataLoader:
    """
    Flexible CSV data loader that can handle various data formats and structures
    for cross-platform user identification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CSV data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Column mapping patterns
        self.column_patterns = {
            'user_id': ['user_id', 'id', 'username', 'user', 'account_id', 'profile_id'],
            'name': ['name', 'full_name', 'display_name', 'real_name', 'first_name'],
            'bio': ['bio', 'description', 'about', 'summary', 'profile_description'],
            'location': ['location', 'place', 'city', 'country', 'address', 'geo'],
            'content': ['content', 'text', 'post', 'message', 'caption', 'tweet'],
            'timestamp': ['timestamp', 'date', 'created_at', 'time', 'posted_at'],
            'likes': ['likes', 'like_count', 'hearts', 'favorites', 'thumbs_up'],
            'followers': ['followers', 'follower_count', 'subscribers', 'connections'],
            'following': ['following', 'following_count', 'friends', 'connections_count'],
            'platform': ['platform', 'source', 'network', 'site', 'origin'],
            'email': ['email', 'email_address', 'mail', 'contact_email']
        }
        
        # Platform detection patterns
        self.platform_patterns = {
            'linkedin': ['linkedin', 'ln', 'professional'],
            'instagram': ['instagram', 'ig', 'insta'],
            'twitter': ['twitter', 'tw', 'tweet'],
            'facebook': ['facebook', 'fb'],
            'tiktok': ['tiktok', 'tt'],
            'youtube': ['youtube', 'yt']
        }
        
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with automatic encoding detection and error handling.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                self.logger.info(f"Successfully loaded {file_path} with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")
                break
        
        raise ValueError(f"Could not load {file_path} with any of the attempted encodings")
    
    def detect_file_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the structure and content type of the CSV file.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing file structure information
        """
        structure = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_mapping': {},
            'data_types': {},
            'platform_detected': None,
            'file_type': 'unknown',
            'quality_score': 0.0
        }
        
        # Map columns to standard names
        structure['column_mapping'] = self._map_columns(df.columns.tolist())
        
        # Detect data types
        structure['data_types'] = self._detect_data_types(df)
        
        # Detect platform
        structure['platform_detected'] = self._detect_platform(df)
        
        # Determine file type (profiles, posts, network, etc.)
        structure['file_type'] = self._detect_file_type(df, structure['column_mapping'])
        
        # Calculate quality score
        structure['quality_score'] = self._calculate_quality_score(df, structure)
        
        return structure
    
    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map original columns to standardized names."""
        column_mapping = {}
        
        for standard_name, patterns in self.column_patterns.items():
            for col in columns:
                col_lower = col.lower().strip()
                if any(pattern in col_lower for pattern in patterns):
                    if standard_name not in column_mapping:
                        column_mapping[standard_name] = col
                    break
        
        return column_mapping
    
    def _detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect and classify data types."""
        data_types = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's a date/time column
                sample_values = df[col].dropna().head(10)
                # Ensure we have a Series for the helper methods
                if isinstance(sample_values, pd.Series) and len(sample_values) > 0:
                    if self._is_datetime_column(sample_values):
                        data_types[col] = 'datetime'
                    elif self._is_url_column(sample_values):
                        data_types[col] = 'url'
                    elif self._is_email_column(sample_values):
                        data_types[col] = 'email'
                    else:
                        data_types[col] = 'text'
                else:
                    data_types[col] = 'text'
            elif pd.api.types.is_numeric_dtype(df[col]):
                data_types[col] = 'numeric'
            else:
                data_types[col] = 'other'
        
        return data_types
    
    def _detect_platform(self, df: pd.DataFrame) -> Optional[str]:
        """Detect which platform the data comes from."""
        # Check if there's a platform column
        platform_cols = [col for col in df.columns 
                        if any(pattern in col.lower() for pattern in ['platform', 'source', 'network'])]
        
        if platform_cols:
            platform_values = df[platform_cols[0]].str.lower().value_counts()
            most_common = platform_values.index[0] if len(platform_values) > 0 else None
            
            for platform, patterns in self.platform_patterns.items():
                if most_common and any(pattern in most_common for pattern in patterns):
                    return platform
        
        # Check column names for platform indicators
        all_columns = ' '.join(df.columns).lower()
        for platform, patterns in self.platform_patterns.items():
            if any(pattern in all_columns for pattern in patterns):
                return platform
        
        # Check data content for platform indicators
        text_columns = df.select_dtypes(include=['object']).columns[:3]  # Check first 3 text columns
        for col in text_columns:
            sample_text = ' '.join(df[col].dropna().head(100).astype(str)).lower()
            for platform, patterns in self.platform_patterns.items():
                if any(pattern in sample_text for pattern in patterns):
                    return platform
        
        return None
    
    def _detect_file_type(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> str:
        """Detect the type of data file (profiles, posts, network, etc.)."""
        
        # Check for network/edge list structure
        if ('source' in column_mapping and 'target' in column_mapping) or \
           (len(df.columns) >= 2 and all(df[col].dtype == 'object' for col in df.columns[:2])):
            return 'network'
        
        # Check for posts/content structure
        if 'content' in column_mapping or 'timestamp' in column_mapping:
            return 'posts'
        
        # Check for profile structure
        if 'bio' in column_mapping or 'name' in column_mapping or 'followers' in column_mapping:
            return 'profiles'
        
        # Check for ground truth structure
        if len(df.columns) >= 3 and any('match' in col.lower() for col in df.columns):
            return 'ground_truth'
        
        return 'unknown'
    
    def _calculate_quality_score(self, df: pd.DataFrame, structure: Dict[str, Any]) -> float:
        """Calculate a quality score for the data."""
        score = 0.0
        
        # Completeness score (40% weight)
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 0.4
        score += completeness
        
        # Column mapping score (30% weight)
        mapped_columns = len(structure['column_mapping']) / max(len(df.columns), 1) * 0.3
        score += mapped_columns
        
        # Data type diversity (20% weight)
        type_diversity = len(set(structure['data_types'].values())) / 4 * 0.2  # Max 4 types
        score += type_diversity
        
        # Platform detection (10% weight)
        platform_score = 0.1 if structure['platform_detected'] else 0
        score += platform_score
        
        return min(score, 1.0)
    
    def _is_datetime_column(self, sample_values: pd.Series) -> bool:
        """Check if a column contains datetime values."""
        try:
            pd.to_datetime(sample_values.head(5))
            return True
        except:
            return False
    
    def _is_url_column(self, sample_values: pd.Series) -> bool:
        """Check if a column contains URLs."""
        url_pattern = re.compile(r'https?://[^\s]+')
        return sample_values.astype(str).str.contains(url_pattern).any()
    
    def _is_email_column(self, sample_values: pd.Series) -> bool:
        """Check if a column contains email addresses."""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return sample_values.astype(str).str.contains(email_pattern).any()
    
    def standardize_dataframe(self, df: pd.DataFrame, structure: Dict[str, Any]) -> pd.DataFrame:
        """
        Standardize the DataFrame to common column names and formats.
        
        Args:
            df: Input DataFrame
            structure: Structure information from detect_file_structure
            
        Returns:
            Standardized DataFrame
        """
        standardized_df = df.copy()
        
        # Rename columns based on mapping
        column_mapping = structure['column_mapping']
        rename_dict = {v: k for k, v in column_mapping.items()}
        standardized_df = standardized_df.rename(columns=rename_dict)
        
        # Standardize data types
        data_types = structure['data_types']
        
        for original_col, dtype in data_types.items():
            # Find the standardized column name
            std_col = rename_dict.get(original_col, original_col)
            
            if std_col in standardized_df.columns:
                if dtype == 'datetime':
                    try:
                        standardized_df[std_col] = pd.to_datetime(standardized_df[std_col])
                    except:
                        pass
                elif dtype == 'numeric':
                    try:
                        standardized_df[std_col] = pd.to_numeric(standardized_df[std_col], errors='coerce')
                    except:
                        pass
        
        # Add platform column if detected and not present
        if structure['platform_detected'] and 'platform' not in standardized_df.columns:
            standardized_df['platform'] = structure['platform_detected']
        
        # Clean text columns
        text_columns = ['name', 'bio', 'content', 'location']
        for col in text_columns:
            if col in standardized_df.columns:
                try:
                    # Convert to string and strip whitespace
                    series = standardized_df[col].astype(str)
                    series = series.str.strip()
                    series = series.replace('nan', np.nan)
                    standardized_df[col] = series
                except Exception as e:
                    self.logger.warning(f"Could not clean column {col}: {e}")
                    # Keep original column if cleaning fails
                    pass
        
        return standardized_df
    
    def load_and_analyze(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load CSV file and perform complete analysis.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (standardized_dataframe, analysis_results)
        """
        # Load the file
        df = self.load_csv(file_path)
        
        # Detect structure
        structure = self.detect_file_structure(df)
        
        # Standardize
        standardized_df = self.standardize_dataframe(df, structure)
        
        # Additional analysis
        analysis = {
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'structure': structure,
            'recommendations': self._generate_recommendations(structure),
            'preview': standardized_df.head().to_dict(),
            'summary_stats': self._generate_summary_stats(standardized_df)
        }
        
        self.logger.info(f"Analysis complete for {file_path}")
        self.logger.info(f"File type: {structure['file_type']}, Platform: {structure['platform_detected']}")
        self.logger.info(f"Quality score: {structure['quality_score']:.2f}")
        
        return standardized_df, analysis
    
    def _generate_recommendations(self, structure: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []
        
        if structure['quality_score'] < 0.7:
            recommendations.append("Consider cleaning the data to improve quality score")
        
        if not structure['platform_detected']:
            recommendations.append("Add a platform indicator column for better analysis")
        
        if len(structure['column_mapping']) < len(structure['data_types']) * 0.5:
            recommendations.append("Review column names - many don't match standard patterns")
        
        if structure['file_type'] == 'unknown':
            recommendations.append("Data structure is unclear - consider restructuring")
        
        return recommendations
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the DataFrame."""
        stats = {
            'total_records': len(df),
            'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 'N/A',
            'date_range': 'N/A',
            'text_columns': [],
            'numeric_columns': []
        }
        
        # Date range
        if 'timestamp' in df.columns:
            try:
                min_date = df['timestamp'].min()
                max_date = df['timestamp'].max()
                stats['date_range'] = f"{min_date} to {max_date}"
            except:
                pass
        
        # Column types
        for col in df.columns:
            if df[col].dtype == 'object':
                stats['text_columns'].append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                stats['numeric_columns'].append(col)
        
        return stats
    
    def batch_load_directory(self, directory_path: str, pattern: str = "*.csv") -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Load and analyze all CSV files in a directory.
        
        Args:
            directory_path: Path to directory containing CSV files
            pattern: File pattern to match (default: "*.csv")
            
        Returns:
            Dictionary mapping filenames to (dataframe, analysis) tuples
        """
        import glob
        
        csv_files = glob.glob(os.path.join(directory_path, pattern))
        results = {}
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            try:
                df, analysis = self.load_and_analyze(file_path)
                results[filename] = (df, analysis)
                self.logger.info(f"Successfully processed {filename}")
            except Exception as e:
                self.logger.error(f"Failed to process {filename}: {str(e)}")
                results[filename] = (None, {'error': str(e)})
        
        return results
    
    def validate_for_cross_platform_analysis(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate loaded dataframes for cross-platform analysis readiness.
        
        Args:
            dataframes: Dictionary of platform name -> DataFrame
            
        Returns:
            Validation results
        """
        validation = {
            'is_ready': False,
            'platforms_detected': [],
            'common_fields': [],
            'issues': [],
            'recommendations': []
        }
        
        if len(dataframes) < 2:
            validation['issues'].append("Need at least 2 platforms for cross-platform analysis")
            return validation
        
        # Check for common fields across platforms
        all_columns = [set(df.columns) for df in dataframes.values() if df is not None]
        if all_columns:
            common_fields = set.intersection(*all_columns)
            validation['common_fields'] = list(common_fields)
        
        # Check for user identifiers
        has_user_ids = all('user_id' in df.columns for df in dataframes.values() if df is not None)
        if not has_user_ids:
            validation['issues'].append("Not all datasets have user_id columns")
        
        # Check for matchable content
        matchable_fields = ['name', 'bio', 'content', 'email', 'location']
        has_matchable = any(field in validation['common_fields'] for field in matchable_fields)
        if not has_matchable:
            validation['issues'].append("No common matchable fields found across platforms")
        
        validation['platforms_detected'] = list(dataframes.keys())
        validation['is_ready'] = len(validation['issues']) == 0
        
        if not validation['is_ready']:
            validation['recommendations'].extend([
                "Ensure all datasets have user_id columns",
                "Include common fields like name, bio, or email for matching",
                "Standardize column names across platforms"
            ])
        
        return validation