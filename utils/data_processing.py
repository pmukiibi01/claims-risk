"""
Data Processing Utilities
Handles claims data processing, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class ClaimsProcessor:
    """
    Processes and cleans claims data for risk adjustment modeling.
    """
    
    def __init__(self):
        self.required_columns = [
            'member_id', 'age', 'gender', 'total_cost', 'total_claims'
        ]
        self.optional_columns = [
            'diabetes', 'hypertension', 'heart_disease', 'copd',
            'cancer', 'kidney_disease', 'mental_health'
        ]
    
    def process_claims(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean claims data.
        """
        logger.info("Processing claims data")
        
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Validate required columns
        missing_columns = [col for col in self.required_columns if col not in processed_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean and validate data
        processed_df = self._clean_data(processed_df)
        
        # Add derived features
        processed_df = self._add_derived_features(processed_df)
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Validate data quality
        self._validate_data_quality(processed_df)
        
        logger.info(f"Claims data processed successfully. Shape: {processed_df.shape}")
        return processed_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the claims data.
        """
        logger.info("Cleaning claims data")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Clean member IDs
        if 'member_id' in df.columns:
            df['member_id'] = df['member_id'].astype(str).str.strip()
        
        # Clean age data
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            # Remove unrealistic ages
            df = df[(df['age'] >= 0) & (df['age'] <= 120)]
        
        # Clean gender data
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.lower().str.strip()
            # Standardize gender values
            gender_mapping = {
                'm': 'male', 'f': 'female', 'male': 'male', 'female': 'female',
                '1': 'male', '0': 'female', 'm': 'male', 'f': 'female'
            }
            df['gender'] = df['gender'].map(gender_mapping).fillna('unknown')
        
        # Clean cost data
        if 'total_cost' in df.columns:
            df['total_cost'] = pd.to_numeric(df['total_cost'], errors='coerce')
            # Remove negative costs
            df = df[df['total_cost'] >= 0]
        
        # Clean claims count
        if 'total_claims' in df.columns:
            df['total_claims'] = pd.to_numeric(df['total_claims'], errors='coerce')
            # Remove negative claims
            df = df[df['total_claims'] >= 0]
        
        # Clean condition flags
        condition_columns = [
            'diabetes', 'hypertension', 'heart_disease', 'copd',
            'cancer', 'kidney_disease', 'mental_health'
        ]
        
        for col in condition_columns:
            if col in df.columns:
                # Convert to binary (0/1)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[col] = (df[col] > 0).astype(int)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for risk adjustment.
        """
        logger.info("Adding derived features")
        
        # Cost per claim
        if 'total_cost' in df.columns and 'total_claims' in df.columns:
            df['cost_per_claim'] = np.where(
                df['total_claims'] > 0,
                df['total_cost'] / df['total_claims'],
                0
            )
        
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 18, 35, 50, 65, 100],
                labels=['0-17', '18-34', '35-49', '50-64', '65+'],
                include_lowest=True
            )
        
        # Chronic condition count
        chronic_conditions = ['diabetes', 'hypertension', 'heart_disease', 'copd']
        available_chronic = [col for col in chronic_conditions if col in df.columns]
        if available_chronic:
            df['chronic_condition_count'] = df[available_chronic].sum(axis=1)
        else:
            df['chronic_condition_count'] = 0
        
        # High-cost condition count
        high_cost_conditions = ['cancer', 'kidney_disease', 'mental_health']
        available_high_cost = [col for col in high_cost_conditions if col in df.columns]
        if available_high_cost:
            df['high_cost_condition_count'] = df[available_high_cost].sum(axis=1)
        else:
            df['high_cost_condition_count'] = 0
        
        # Total condition count
        all_conditions = available_chronic + available_high_cost
        if all_conditions:
            df['total_condition_count'] = df[all_conditions].sum(axis=1)
        else:
            df['total_condition_count'] = 0
        
        # Utilization flags
        if 'total_claims' in df.columns:
            # High utilizer (top 20% of claims)
            claims_threshold = df['total_claims'].quantile(0.8)
            df['high_utilizer'] = (df['total_claims'] >= claims_threshold).astype(int)
        
        if 'total_cost' in df.columns:
            # High cost member (top 20% of costs)
            cost_threshold = df['total_cost'].quantile(0.8)
            df['high_cost_member'] = (df['total_cost'] >= cost_threshold).astype(int)
        
        # Risk flags
        if 'age' in df.columns:
            df['elderly'] = (df['age'] >= 65).astype(int)
            df['young_adult'] = ((df['age'] >= 18) & (df['age'] <= 35)).astype(int)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        logger.info("Handling missing values")
        
        # Fill missing values for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill missing values for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame):
        """
        Validate data quality and log issues.
        """
        logger.info("Validating data quality")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check for negative values in cost/claims
        if 'total_cost' in df.columns:
            negative_costs = (df['total_cost'] < 0).sum()
            if negative_costs > 0:
                logger.warning(f"Found {negative_costs} records with negative costs")
        
        if 'total_claims' in df.columns:
            negative_claims = (df['total_claims'] < 0).sum()
            if negative_claims > 0:
                logger.warning(f"Found {negative_claims} records with negative claims")
        
        # Check for unrealistic ages
        if 'age' in df.columns:
            unrealistic_ages = ((df['age'] < 0) | (df['age'] > 120)).sum()
            if unrealistic_ages > 0:
                logger.warning(f"Found {unrealistic_ages} records with unrealistic ages")
        
        # Check data distribution
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Unique members: {df['member_id'].nunique() if 'member_id' in df.columns else 'N/A'}")
        
        if 'total_cost' in df.columns:
            logger.info(f"Cost statistics: mean={df['total_cost'].mean():.2f}, "
                       f"median={df['total_cost'].median():.2f}, "
                       f"std={df['total_cost'].std():.2f}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the processed data.
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary['categorical_summary'] = {}
            for col in categorical_cols:
                summary['categorical_summary'][col] = df[col].value_counts().to_dict()
        
        return summary