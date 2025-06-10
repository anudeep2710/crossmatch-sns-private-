"""
Privacy protection module for GDPR/CCPA compliance and differential privacy.
"""

import numpy as np
import pandas as pd
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import warnings

class PrivacyProtector:
    """
    Privacy protection module implementing differential privacy, data anonymization,
    and GDPR/CCPA compliance features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize privacy protector.
        
        Args:
            config: Configuration dictionary containing privacy parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Differential privacy parameters
        self.epsilon = config.get('dp_epsilon', 1.0)
        self.delta = config.get('dp_delta', 1e-5)
        self.sensitivity = config.get('dp_sensitivity', 1.0)
        
        # Anonymization parameters
        self.k_anonymity = config.get('k_anonymity', 5)
        self.l_diversity = config.get('l_diversity', 3)
        
        # Encryption key for sensitive data
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Data retention policy
        self.data_retention_days = config.get('data_retention_days', 365)
        
        # Audit logging
        self.audit_log = []
        self.enable_audit_logging = config.get('enable_audit_logging', True)
        
        # Consent management
        self.consent_records = {}
        
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive data."""
        password = self.config.get('encryption_password', 'default_password').encode()
        salt = self.config.get('encryption_salt', 'default_salt').encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def add_differential_privacy_noise(self, data: np.ndarray, 
                                     sensitivity: Optional[float] = None) -> np.ndarray:
        """
        Add differential privacy noise to numerical data.
        
        Args:
            data: Input data array
            sensitivity: Sensitivity parameter (defaults to class sensitivity)
            
        Returns:
            Data with added noise
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        # Laplace mechanism for differential privacy
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        
        noisy_data = data + noise
        
        self._log_privacy_action(
            action="differential_privacy_noise_added",
            details={
                "epsilon": self.epsilon,
                "delta": self.delta,
                "sensitivity": sensitivity,
                "data_shape": data.shape
            }
        )
        
        return noisy_data
    
    def anonymize_identifiers(self, data: pd.DataFrame, 
                            identifier_columns: List[str]) -> pd.DataFrame:
        """
        Anonymize direct identifiers using hashing.
        
        Args:
            data: Input dataframe
            identifier_columns: List of columns containing direct identifiers
            
        Returns:
            Dataframe with anonymized identifiers
        """
        anonymized_data = data.copy()
        
        for column in identifier_columns:
            if column in anonymized_data.columns:
                anonymized_data[column] = anonymized_data[column].apply(
                    lambda x: self._hash_identifier(str(x)) if pd.notna(x) else x
                )
        
        self._log_privacy_action(
            action="identifiers_anonymized",
            details={
                "columns": identifier_columns,
                "num_records": len(data)
            }
        )
        
        return anonymized_data
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash an identifier using SHA-256."""
        salt = self.config.get('hash_salt', 'privacy_salt').encode()
        hasher = hashlib.sha256()
        hasher.update(salt + identifier.encode())
        return hasher.hexdigest()[:16]  # Truncate for readability
    
    def apply_k_anonymity(self, data: pd.DataFrame, 
                         quasi_identifiers: List[str]) -> pd.DataFrame:
        """
        Apply k-anonymity by generalizing quasi-identifiers.
        
        Args:
            data: Input dataframe
            quasi_identifiers: List of quasi-identifier columns
            
        Returns:
            K-anonymous dataframe
        """
        k_anon_data = data.copy()
        
        # Group by quasi-identifiers and check group sizes
        grouped = k_anon_data.groupby(quasi_identifiers)
        small_groups = grouped.filter(lambda x: len(x) < self.k_anonymity)
        
        if len(small_groups) > 0:
            # Generalize small groups
            k_anon_data = self._generalize_small_groups(
                k_anon_data, quasi_identifiers, small_groups
            )
        
        self._log_privacy_action(
            action="k_anonymity_applied",
            details={
                "k": self.k_anonymity,
                "quasi_identifiers": quasi_identifiers,
                "original_groups": len(grouped),
                "small_groups_count": len(small_groups)
            }
        )
        
        return k_anon_data
    
    def _generalize_small_groups(self, data: pd.DataFrame, 
                               quasi_identifiers: List[str],
                               small_groups: pd.DataFrame) -> pd.DataFrame:
        """Generalize quasi-identifiers for small groups."""
        generalized_data = data.copy()
        
        for column in quasi_identifiers:
            if column in data.columns:
                if pd.api.types.is_numeric_dtype(data[column]):
                    # Numerical generalization: use ranges
                    generalized_data[column] = self._generalize_numerical(
                        data[column], small_groups.index
                    )
                else:
                    # Categorical generalization: use broader categories
                    generalized_data[column] = self._generalize_categorical(
                        data[column], small_groups.index
                    )
        
        return generalized_data
    
    def _generalize_numerical(self, series: pd.Series, indices: pd.Index) -> pd.Series:
        """Generalize numerical values to ranges."""
        generalized = series.copy()
        
        for idx in indices:
            value = series.iloc[idx]
            if pd.notna(value):
                # Create range buckets
                bucket_size = max(1, value * 0.1)  # 10% range
                lower = int(value - bucket_size)
                upper = int(value + bucket_size)
                generalized.iloc[idx] = f"{lower}-{upper}"
        
        return generalized
    
    def _generalize_categorical(self, series: pd.Series, indices: pd.Index) -> pd.Series:
        """Generalize categorical values."""
        generalized = series.copy()
        
        for idx in indices:
            value = series.iloc[idx]
            if pd.notna(value):
                # Simple generalization: use "Other" category
                generalized.iloc[idx] = "Other"
        
        return generalized
    
    def encrypt_sensitive_data(self, data: Union[str, Dict, List]) -> str:
        """
        Encrypt sensitive data using Fernet encryption.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        # Convert to JSON string if not already string
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        # Encrypt
        encrypted_data = self.cipher_suite.encrypt(data_str.encode())
        
        self._log_privacy_action(
            action="data_encrypted",
            details={
                "data_type": type(data).__name__,
                "encrypted_size": len(encrypted_data)
            }
        )
        
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Any:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            # Decode and decrypt
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
            decrypted_str = decrypted_bytes.decode()
            
            # Try to parse as JSON, fall back to string
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str
                
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")
    
    def record_consent(self, user_id: str, consent_type: str, 
                      granted: bool, purpose: str) -> str:
        """
        Record user consent for data processing.
        
        Args:
            user_id: User identifier
            consent_type: Type of consent (e.g., 'data_collection', 'data_sharing')
            granted: Whether consent was granted
            purpose: Purpose for which consent is requested
            
        Returns:
            Consent record ID
        """
        consent_id = secrets.token_hex(16)
        timestamp = datetime.now().isoformat()
        
        consent_record = {
            'consent_id': consent_id,
            'user_id': self._hash_identifier(user_id),
            'consent_type': consent_type,
            'granted': granted,
            'purpose': purpose,
            'timestamp': timestamp,
            'expiry': (datetime.now() + timedelta(days=365)).isoformat()
        }
        
        self.consent_records[consent_id] = consent_record
        
        self._log_privacy_action(
            action="consent_recorded",
            details={
                "consent_id": consent_id,
                "consent_type": consent_type,
                "granted": granted,
                "purpose": purpose
            }
        )
        
        return consent_id
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """
        Check if user has valid consent for a specific purpose.
        
        Args:
            user_id: User identifier
            purpose: Purpose to check consent for
            
        Returns:
            Whether valid consent exists
        """
        hashed_user_id = self._hash_identifier(user_id)
        current_time = datetime.now()
        
        for record in self.consent_records.values():
            if (record['user_id'] == hashed_user_id and 
                record['purpose'] == purpose and 
                record['granted'] and
                datetime.fromisoformat(record['expiry']) > current_time):
                return True
        
        return False
    
    def anonymize_embeddings(self, embeddings: np.ndarray, 
                           privacy_budget: float = 1.0) -> np.ndarray:
        """
        Anonymize embeddings using differential privacy.
        
        Args:
            embeddings: Input embeddings
            privacy_budget: Privacy budget (epsilon)
            
        Returns:
            Anonymized embeddings
        """
        # Temporarily adjust epsilon for this operation
        original_epsilon = self.epsilon
        self.epsilon = privacy_budget
        
        # Add noise to embeddings
        noisy_embeddings = self.add_differential_privacy_noise(embeddings)
        
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        # Clip embeddings to maintain utility
        clipped_embeddings = np.clip(noisy_embeddings, -3, 3)
        
        self._log_privacy_action(
            action="embeddings_anonymized",
            details={
                "input_shape": embeddings.shape,
                "privacy_budget": privacy_budget,
                "clipping_range": (-3, 3)
            }
        )
        
        return clipped_embeddings
    
    def secure_multiparty_computation(self, embeddings1: np.ndarray, 
                                    embeddings2: np.ndarray) -> np.ndarray:
        """
        Perform secure multiparty computation for similarity calculation.
        
        Args:
            embeddings1: Embeddings from party 1
            embeddings2: Embeddings from party 2
            
        Returns:
            Privately computed similarities
        """
        # Simple SMPC simulation using additive secret sharing
        
        # Split embeddings1 into random shares
        share1_1 = np.random.random(embeddings1.shape)
        share1_2 = embeddings1 - share1_1
        
        # Split embeddings2 into random shares
        share2_1 = np.random.random(embeddings2.shape)
        share2_2 = embeddings2 - share2_1
        
        # Compute dot product shares (simplified)
        result_share1 = np.sum(share1_1 * share2_1, axis=1)
        result_share2 = np.sum(share1_2 * share2_2, axis=1)
        
        # Combine shares to get final result
        similarities = result_share1 + result_share2
        
        # Add differential privacy noise
        similarities = self.add_differential_privacy_noise(similarities)
        
        self._log_privacy_action(
            action="smpc_computation",
            details={
                "embeddings1_shape": embeddings1.shape,
                "embeddings2_shape": embeddings2.shape,
                "result_shape": similarities.shape
            }
        )
        
        return similarities
    
    def data_minimization(self, data: pd.DataFrame, 
                         essential_columns: List[str]) -> pd.DataFrame:
        """
        Apply data minimization by keeping only essential columns.
        
        Args:
            data: Input dataframe
            essential_columns: List of essential columns to keep
            
        Returns:
            Minimized dataframe
        """
        available_columns = [col for col in essential_columns if col in data.columns]
        minimized_data = data[available_columns].copy()
        
        self._log_privacy_action(
            action="data_minimization",
            details={
                "original_columns": len(data.columns),
                "minimized_columns": len(minimized_data.columns),
                "removed_columns": len(data.columns) - len(minimized_data.columns)
            }
        )
        
        return minimized_data
    
    def apply_retention_policy(self, data: pd.DataFrame, 
                             timestamp_column: str) -> pd.DataFrame:
        """
        Apply data retention policy by removing old data.
        
        Args:
            data: Input dataframe
            timestamp_column: Column containing timestamps
            
        Returns:
            Filtered dataframe with recent data only
        """
        if timestamp_column not in data.columns:
            self.logger.warning(f"Timestamp column '{timestamp_column}' not found")
            return data
        
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        
        # Convert timestamp column to datetime if needed
        timestamp_series = pd.to_datetime(data[timestamp_column])
        
        # Filter recent data
        recent_data = data[timestamp_series >= cutoff_date].copy()
        
        self._log_privacy_action(
            action="retention_policy_applied",
            details={
                "retention_days": self.data_retention_days,
                "original_records": len(data),
                "retained_records": len(recent_data),
                "removed_records": len(data) - len(recent_data)
            }
        )
        
        return recent_data
    
    def _log_privacy_action(self, action: str, details: Dict[str, Any]):
        """Log privacy-related actions for audit purposes."""
        if not self.enable_audit_logging:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        
        self.audit_log.append(log_entry)
        
        # Keep only recent audit logs
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def get_audit_log(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            since: Only return entries since this timestamp
            
        Returns:
            List of audit log entries
        """
        if since is None:
            return self.audit_log.copy()
        
        filtered_log = []
        for entry in self.audit_log:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if entry_time >= since:
                filtered_log.append(entry)
        
        return filtered_log
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """
        Generate privacy compliance report.
        
        Returns:
            Privacy report dictionary
        """
        total_consents = len(self.consent_records)
        granted_consents = sum(1 for r in self.consent_records.values() if r['granted'])
        
        recent_actions = self.get_audit_log(
            since=datetime.now() - timedelta(days=30)
        )
        
        action_counts = {}
        for entry in recent_actions:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'privacy_parameters': {
                'epsilon': self.epsilon,
                'delta': self.delta,
                'k_anonymity': self.k_anonymity,
                'data_retention_days': self.data_retention_days
            },
            'consent_summary': {
                'total_consents': total_consents,
                'granted_consents': granted_consents,
                'consent_rate': granted_consents / total_consents if total_consents > 0 else 0
            },
            'recent_actions': action_counts,
            'compliance_status': {
                'gdpr_compliant': self._check_gdpr_compliance(),
                'ccpa_compliant': self._check_ccpa_compliance()
            }
        }
        
        return report
    
    def _check_gdpr_compliance(self) -> bool:
        """Check GDPR compliance status."""
        # Basic compliance checks
        checks = [
            self.enable_audit_logging,  # Audit logging enabled
            len(self.consent_records) > 0,  # Consent management in place
            self.data_retention_days <= 365,  # Reasonable retention policy
            self.epsilon <= 2.0  # Reasonable privacy parameters
        ]
        
        return all(checks)
    
    def _check_ccpa_compliance(self) -> bool:
        """Check CCPA compliance status."""
        # Basic compliance checks (similar to GDPR for this implementation)
        return self._check_gdpr_compliance()
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """Get privacy protection statistics."""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'k_anonymity': self.k_anonymity,
            'total_consents': len(self.consent_records),
            'audit_log_entries': len(self.audit_log),
            'encryption_enabled': True,
            'retention_days': self.data_retention_days
        }