"""
Test suite for the Claims-Based Risk Adjustment System
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import tempfile
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app
from models.risk_adjustment import RiskAdjustmentModel
from models.cost_drivers import CostDriverAnalysis
from utils.data_processing import ClaimsProcessor
from utils.hcc_mapping import HCCMapper
from data.sample_data import generate_sample_data

client = TestClient(app)

class TestRiskAdjustmentModel:
    """Test the risk adjustment model functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = generate_sample_data(n_members=100, n_claims=500)
        self.model = RiskAdjustmentModel()
    
    def test_prepare_features(self):
        """Test feature preparation."""
        features = self.model.prepare_features(self.sample_data)
        
        # Check that features are created
        assert len(features) == len(self.sample_data)
        assert 'age' in features.columns
        assert 'hcc_score' in features.columns
        assert 'chronic_conditions' in features.columns
    
    def test_train_model(self):
        """Test model training."""
        model_results = self.model.train(self.sample_data)
        
        # Check that model was trained
        assert self.model.model is not None
        assert 'best_model' in model_results
        assert 'best_r2' in model_results
        assert model_results['best_r2'] >= 0
    
    def test_predict_member_risk(self):
        """Test member risk prediction."""
        # Train model first
        self.model.train(self.sample_data)
        
        # Test prediction
        member_data = self.sample_data.iloc[0:1]
        risk_score = self.model.predict_member_risk(member_data)
        
        assert isinstance(risk_score, float)
        assert risk_score >= 0
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Train model first
        self.model.train(self.sample_data)
        
        importance = self.model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0

class TestCostDriverAnalysis:
    """Test the cost driver analysis functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = generate_sample_data(n_members=100, n_claims=500)
        self.cost_analyzer = CostDriverAnalysis()
    
    def test_prepare_cost_analysis_data(self):
        """Test cost analysis data preparation."""
        X, y = self.cost_analyzer._prepare_cost_analysis_data(self.sample_data)
        
        assert len(X) == len(y)
        assert 'age' in X.columns
        assert 'total_cost' in y.name or 'total_cost' in y.index.name
    
    def test_analyze_cost_drivers(self):
        """Test cost driver analysis."""
        results = self.cost_analyzer.analyze(self.sample_data)
        
        assert 'cost_drivers' in results
        assert 'shapley_values' in results
        assert 'insights' in results
        assert 'variation_metrics' in results
    
    def test_identify_cost_drivers(self):
        """Test cost driver identification."""
        X, y = self.cost_analyzer._prepare_cost_analysis_data(self.sample_data)
        self.cost_analyzer._train_cost_model(X, y)
        
        cost_drivers = self.cost_analyzer._identify_cost_drivers(X, y)
        
        assert 'feature_importance' in cost_drivers
        assert 'correlations' in cost_drivers
        assert 'top_drivers' in cost_drivers

class TestClaimsProcessor:
    """Test the claims data processor."""
    
    def setup_method(self):
        """Set up test data."""
        self.processor = ClaimsProcessor()
        self.sample_data = generate_sample_data(n_members=50, n_claims=200)
    
    def test_process_claims(self):
        """Test claims processing."""
        processed_data = self.processor.process_claims(self.sample_data)
        
        # Check that data was processed
        assert len(processed_data) > 0
        assert 'member_id' in processed_data.columns
        assert 'total_cost' in processed_data.columns
    
    def test_clean_data(self):
        """Test data cleaning."""
        # Add some dirty data
        dirty_data = self.sample_data.copy()
        dirty_data.loc[0, 'age'] = -5  # Invalid age
        dirty_data.loc[1, 'total_cost'] = -100  # Negative cost
        
        cleaned_data = self.processor._clean_data(dirty_data)
        
        # Check that invalid data was removed or fixed
        assert (cleaned_data['age'] >= 0).all()
        assert (cleaned_data['total_cost'] >= 0).all()
    
    def test_add_derived_features(self):
        """Test derived feature creation."""
        features_data = self.processor._add_derived_features(self.sample_data)
        
        # Check that derived features were created
        assert 'cost_per_claim' in features_data.columns
        assert 'age_group' in features_data.columns
        assert 'chronic_condition_count' in features_data.columns

class TestHCCMapper:
    """Test the HCC mapping functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.hcc_mapper = HCCMapper()
    
    def test_map_conditions_to_hcc(self):
        """Test condition to HCC mapping."""
        conditions = {
            'diabetes': 1,
            'hypertension': 1,
            'cancer': 0
        }
        
        hcc_codes = self.hcc_mapper.map_conditions_to_hcc(conditions)
        
        assert isinstance(hcc_codes, list)
        assert len(hcc_codes) > 0
    
    def test_calculate_hcc_score(self):
        """Test HCC score calculation."""
        conditions = {
            'diabetes': 1,
            'hypertension': 1,
            'cancer': 0
        }
        
        score = self.hcc_mapper.calculate_hcc_score(conditions, 65, 'male')
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_calculate_member_risk_score(self):
        """Test member risk score calculation."""
        member_data = pd.Series({
            'age': 65,
            'gender': 'male',
            'diabetes': 1,
            'hypertension': 1,
            'cancer': 0
        })
        
        risk_score = self.hcc_mapper.calculate_member_risk_score(member_data)
        
        assert isinstance(risk_score, float)
        assert risk_score >= 0

class TestAPIEndpoints:
    """Test the FastAPI endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_root_endpoint(self):
        """Test root endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_generate_sample_data(self):
        """Test sample data generation endpoint."""
        response = client.post("/generate-sample-data")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "records" in data
    
    def test_download_sample(self):
        """Test sample data download endpoint."""
        # First generate sample data
        client.post("/generate-sample-data")
        
        response = client.get("/download-sample")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
    
    def test_train_model_without_data(self):
        """Test model training without data should fail."""
        response = client.post("/train-risk-model")
        assert response.status_code == 404
    
    def test_analyze_cost_drivers_without_data(self):
        """Test cost driver analysis without data should fail."""
        response = client.post("/analyze-cost-drivers")
        assert response.status_code == 404
    
    def test_model_performance_without_model(self):
        """Test model performance without trained model should fail."""
        response = client.get("/model-performance")
        assert response.status_code == 404

class TestSampleDataGenerator:
    """Test the sample data generator."""
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        data = generate_sample_data(n_members=50, n_claims=200)
        
        # Check basic structure
        assert len(data) > 0
        assert 'member_id' in data.columns
        assert 'age' in data.columns
        assert 'total_cost' in data.columns
    
    def test_data_quality(self):
        """Test generated data quality."""
        data = generate_sample_data(n_members=100, n_claims=500)
        
        # Check for required columns
        required_columns = ['member_id', 'age', 'gender', 'total_cost', 'total_claims']
        for col in required_columns:
            assert col in data.columns
        
        # Check data ranges
        assert (data['age'] >= 0).all()
        assert (data['age'] <= 120).all()
        assert (data['total_cost'] >= 0).all()
        assert (data['total_claims'] >= 0).all()
        
        # Check gender values
        assert set(data['gender'].unique()).issubset({'male', 'female'})

def test_integration():
    """Integration test for the complete workflow."""
    # Generate sample data
    data = generate_sample_data(n_members=100, n_claims=500)
    
    # Process data
    processor = ClaimsProcessor()
    processed_data = processor.process_claims(data)
    
    # Train risk model
    risk_model = RiskAdjustmentModel()
    model_results = risk_model.train(processed_data)
    
    # Analyze cost drivers
    cost_analyzer = CostDriverAnalysis()
    cost_results = cost_analyzer.analyze(processed_data)
    
    # Verify results
    assert model_results['best_r2'] >= 0
    assert 'cost_drivers' in cost_results
    assert 'insights' in cost_results

if __name__ == "__main__":
    pytest.main([__file__, "-v"])