"""
HCC (Hierarchical Condition Category) Mapping Utilities
Simplified HCC mapping for risk adjustment modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class HCCMapper:
    """
    Maps clinical conditions to HCC codes and calculates risk scores.
    """
    
    def __init__(self):
        # Simplified HCC mapping (in practice, this would be much more comprehensive)
        self.hcc_mapping = {
            # Diabetes and related conditions
            'diabetes': ['HCC_18', 'HCC_19'],
            'diabetes_complications': ['HCC_17', 'HCC_18', 'HCC_19'],
            
            # Cardiovascular conditions
            'hypertension': ['HCC_85', 'HCC_86'],
            'heart_disease': ['HCC_85', 'HCC_88', 'HCC_96'],
            'heart_failure': ['HCC_85', 'HCC_88'],
            'stroke': ['HCC_100', 'HCC_101'],
            
            # Respiratory conditions
            'copd': ['HCC_111', 'HCC_112'],
            'asthma': ['HCC_111'],
            
            # Cancer conditions
            'cancer': ['HCC_8', 'HCC_9', 'HCC_10', 'HCC_11', 'HCC_12'],
            'breast_cancer': ['HCC_9'],
            'lung_cancer': ['HCC_8'],
            'prostate_cancer': ['HCC_10'],
            
            # Kidney conditions
            'kidney_disease': ['HCC_136', 'HCC_137'],
            'dialysis': ['HCC_136'],
            
            # Mental health conditions
            'mental_health': ['HCC_57', 'HCC_58', 'HCC_59'],
            'depression': ['HCC_58'],
            'anxiety': ['HCC_57'],
            'bipolar': ['HCC_59'],
            
            # Other conditions
            'arthritis': ['HCC_40', 'HCC_41'],
            'osteoporosis': ['HCC_40'],
            'alzheimer': ['HCC_51', 'HCC_52'],
            'parkinson': ['HCC_51'],
            'hiv': ['HCC_1', 'HCC_2'],
            'substance_abuse': ['HCC_54', 'HCC_55']
        }
        
        # HCC weights (simplified - in practice these come from CMS)
        self.hcc_weights = {
            'HCC_1': 1.0,   # HIV/AIDS
            'HCC_2': 0.8,    # HIV/AIDS with complications
            'HCC_8': 2.5,    # Lung cancer
            'HCC_9': 1.8,    # Breast cancer
            'HCC_10': 1.5,   # Prostate cancer
            'HCC_11': 2.0,   # Colorectal cancer
            'HCC_12': 2.2,   # Other cancer
            'HCC_17': 0.5,   # Diabetes without complications
            'HCC_18': 1.2,   # Diabetes with complications
            'HCC_19': 1.5,   # Diabetes with renal complications
            'HCC_40': 0.3,   # Rheumatoid arthritis
            'HCC_41': 0.4,   # Osteoarthritis
            'HCC_51': 1.8,   # Alzheimer's disease
            'HCC_52': 1.5,   # Parkinson's disease
            'HCC_54': 0.6,   # Substance abuse
            'HCC_55': 0.8,   # Substance abuse with complications
            'HCC_57': 0.4,   # Anxiety disorders
            'HCC_58': 0.5,   # Depression
            'HCC_59': 0.7,   # Bipolar disorder
            'HCC_85': 0.8,   # Congestive heart failure
            'HCC_86': 0.6,   # Hypertension
            'HCC_88': 1.2,   # Coronary artery disease
            'HCC_96': 1.0,   # Other heart disease
            'HCC_100': 1.5,  # Stroke
            'HCC_101': 1.8,  # Stroke with complications
            'HCC_111': 0.7,  # COPD
            'HCC_112': 1.0,  # COPD with complications
            'HCC_136': 2.0,  # End-stage renal disease
            'HCC_137': 1.5   # Chronic kidney disease
        }
        
        # Age-sex coefficients (simplified)
        self.age_sex_coefficients = {
            ('male', 0): 0.0,
            ('male', 1): 0.0,
            ('male', 2): 0.0,
            ('male', 3): 0.0,
            ('male', 4): 0.0,
            ('male', 5): 0.0,
            ('male', 6): 0.0,
            ('male', 7): 0.0,
            ('male', 8): 0.0,
            ('male', 9): 0.0,
            ('male', 10): 0.0,
            ('male', 11): 0.0,
            ('male', 12): 0.0,
            ('male', 13): 0.0,
            ('male', 14): 0.0,
            ('male', 15): 0.0,
            ('male', 16): 0.0,
            ('male', 17): 0.0,
            ('male', 18): 0.0,
            ('male', 19): 0.0,
            ('male', 20): 0.0,
            ('male', 21): 0.0,
            ('male', 22): 0.0,
            ('male', 23): 0.0,
            ('male', 24): 0.0,
            ('male', 25): 0.0,
            ('male', 26): 0.0,
            ('male', 27): 0.0,
            ('male', 28): 0.0,
            ('male', 29): 0.0,
            ('male', 30): 0.0,
            ('male', 31): 0.0,
            ('male', 32): 0.0,
            ('male', 33): 0.0,
            ('male', 34): 0.0,
            ('male', 35): 0.0,
            ('male', 36): 0.0,
            ('male', 37): 0.0,
            ('male', 38): 0.0,
            ('male', 39): 0.0,
            ('male', 40): 0.0,
            ('male', 41): 0.0,
            ('male', 42): 0.0,
            ('male', 43): 0.0,
            ('male', 44): 0.0,
            ('male', 45): 0.0,
            ('male', 46): 0.0,
            ('male', 47): 0.0,
            ('male', 48): 0.0,
            ('male', 49): 0.0,
            ('male', 50): 0.0,
            ('male', 51): 0.0,
            ('male', 52): 0.0,
            ('male', 53): 0.0,
            ('male', 54): 0.0,
            ('male', 55): 0.0,
            ('male', 56): 0.0,
            ('male', 57): 0.0,
            ('male', 58): 0.0,
            ('male', 59): 0.0,
            ('male', 60): 0.0,
            ('male', 61): 0.0,
            ('male', 62): 0.0,
            ('male', 63): 0.0,
            ('male', 64): 0.0,
            ('male', 65): 0.0,
            ('male', 66): 0.0,
            ('male', 67): 0.0,
            ('male', 68): 0.0,
            ('male', 69): 0.0,
            ('male', 70): 0.0,
            ('male', 71): 0.0,
            ('male', 72): 0.0,
            ('male', 73): 0.0,
            ('male', 74): 0.0,
            ('male', 75): 0.0,
            ('male', 76): 0.0,
            ('male', 77): 0.0,
            ('male', 78): 0.0,
            ('male', 79): 0.0,
            ('male', 80): 0.0,
            ('male', 81): 0.0,
            ('male', 82): 0.0,
            ('male', 83): 0.0,
            ('male', 84): 0.0,
            ('male', 85): 0.0,
            ('male', 86): 0.0,
            ('male', 87): 0.0,
            ('male', 88): 0.0,
            ('male', 89): 0.0,
            ('male', 90): 0.0,
            ('male', 91): 0.0,
            ('male', 92): 0.0,
            ('male', 93): 0.0,
            ('male', 94): 0.0,
            ('male', 95): 0.0,
            ('male', 96): 0.0,
            ('male', 97): 0.0,
            ('male', 98): 0.0,
            ('male', 99): 0.0,
            ('male', 100): 0.0,
            ('female', 0): 0.0,
            ('female', 1): 0.0,
            ('female', 2): 0.0,
            ('female', 3): 0.0,
            ('female', 4): 0.0,
            ('female', 5): 0.0,
            ('female', 6): 0.0,
            ('female', 7): 0.0,
            ('female', 8): 0.0,
            ('female', 9): 0.0,
            ('female', 10): 0.0,
            ('female', 11): 0.0,
            ('female', 12): 0.0,
            ('female', 13): 0.0,
            ('female', 14): 0.0,
            ('female', 15): 0.0,
            ('female', 16): 0.0,
            ('female', 17): 0.0,
            ('female', 18): 0.0,
            ('female', 19): 0.0,
            ('female', 20): 0.0,
            ('female', 21): 0.0,
            ('female', 22): 0.0,
            ('female', 23): 0.0,
            ('female', 24): 0.0,
            ('female', 25): 0.0,
            ('female', 26): 0.0,
            ('female', 27): 0.0,
            ('female', 28): 0.0,
            ('female', 29): 0.0,
            ('female', 30): 0.0,
            ('female', 31): 0.0,
            ('female', 32): 0.0,
            ('female', 33): 0.0,
            ('female', 34): 0.0,
            ('female', 35): 0.0,
            ('female', 36): 0.0,
            ('female', 37): 0.0,
            ('female', 38): 0.0,
            ('female', 39): 0.0,
            ('female', 40): 0.0,
            ('female', 41): 0.0,
            ('female', 42): 0.0,
            ('female', 43): 0.0,
            ('female', 44): 0.0,
            ('female', 45): 0.0,
            ('female', 46): 0.0,
            ('female', 47): 0.0,
            ('female', 48): 0.0,
            ('female', 49): 0.0,
            ('female', 50): 0.0,
            ('female', 51): 0.0,
            ('female', 52): 0.0,
            ('female', 53): 0.0,
            ('female', 54): 0.0,
            ('female', 55): 0.0,
            ('female', 56): 0.0,
            ('female', 57): 0.0,
            ('female', 58): 0.0,
            ('female', 59): 0.0,
            ('female', 60): 0.0,
            ('female', 61): 0.0,
            ('female', 62): 0.0,
            ('female', 63): 0.0,
            ('female', 64): 0.0,
            ('female', 65): 0.0,
            ('female', 66): 0.0,
            ('female', 67): 0.0,
            ('female', 68): 0.0,
            ('female', 69): 0.0,
            ('female', 70): 0.0,
            ('female', 71): 0.0,
            ('female', 72): 0.0,
            ('female', 73): 0.0,
            ('female', 74): 0.0,
            ('female', 75): 0.0,
            ('female', 76): 0.0,
            ('female', 77): 0.0,
            ('female', 78): 0.0,
            ('female', 79): 0.0,
            ('female', 80): 0.0,
            ('female', 81): 0.0,
            ('female', 82): 0.0,
            ('female', 83): 0.0,
            ('female', 84): 0.0,
            ('female', 85): 0.0,
            ('female', 86): 0.0,
            ('female', 87): 0.0,
            ('female', 88): 0.0,
            ('female', 89): 0.0,
            ('female', 90): 0.0,
            ('female', 91): 0.0,
            ('female', 92): 0.0,
            ('female', 93): 0.0,
            ('female', 94): 0.0,
            ('female', 95): 0.0,
            ('female', 96): 0.0,
            ('female', 97): 0.0,
            ('female', 98): 0.0,
            ('female', 99): 0.0,
            ('female', 100): 0.0
        }
    
    def map_conditions_to_hcc(self, conditions: Dict[str, int]) -> List[str]:
        """
        Map clinical conditions to HCC codes.
        """
        hcc_codes = []
        
        for condition, present in conditions.items():
            if present and condition in self.hcc_mapping:
                hcc_codes.extend(self.hcc_mapping[condition])
        
        return list(set(hcc_codes))  # Remove duplicates
    
    def calculate_hcc_score(self, conditions: Dict[str, int], age: int, gender: str) -> float:
        """
        Calculate HCC risk score for a member.
        """
        # Map conditions to HCC codes
        hcc_codes = self.map_conditions_to_hcc(conditions)
        
        # Calculate base HCC score
        hcc_score = 0.0
        for hcc_code in hcc_codes:
            if hcc_code in self.hcc_weights:
                hcc_score += self.hcc_weights[hcc_code]
        
        # Apply age-sex adjustment
        age_group = min(age // 1, 100)  # Group ages into 1-year buckets
        gender_key = gender.lower() if gender else 'unknown'
        
        if (gender_key, age_group) in self.age_sex_coefficients:
            age_sex_coeff = self.age_sex_coefficients[(gender_key, age_group)]
        else:
            age_sex_coeff = 0.0
        
        # Calculate final risk score
        final_score = hcc_score + age_sex_coeff
        
        return max(0.0, final_score)  # Ensure non-negative score
    
    def get_hcc_weights(self) -> Dict[str, float]:
        """
        Get HCC weights for all codes.
        """
        return self.hcc_weights.copy()
    
    def get_condition_mapping(self) -> Dict[str, List[str]]:
        """
        Get condition to HCC mapping.
        """
        return self.hcc_mapping.copy()
    
    def calculate_member_risk_score(self, member_data: pd.Series) -> float:
        """
        Calculate risk score for a single member.
        """
        # Extract conditions
        condition_columns = [
            'diabetes', 'hypertension', 'heart_disease', 'copd',
            'cancer', 'kidney_disease', 'mental_health'
        ]
        
        conditions = {}
        for col in condition_columns:
            if col in member_data:
                conditions[col] = int(member_data[col]) if not pd.isna(member_data[col]) else 0
            else:
                conditions[col] = 0
        
        # Extract demographics
        age = int(member_data.get('age', 0)) if not pd.isna(member_data.get('age', 0)) else 0
        gender = str(member_data.get('gender', 'unknown')).lower()
        
        # Calculate HCC score
        hcc_score = self.calculate_hcc_score(conditions, age, gender)
        
        return hcc_score