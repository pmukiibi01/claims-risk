#!/usr/bin/env python3
"""
Basic test script for the Claims-Based Risk Adjustment System
Tests file structure and basic functionality without external dependencies.
"""

import sys
import os
from datetime import datetime

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        '.gitignore',
        'models/risk_adjustment.py',
        'models/cost_drivers.py',
        'utils/data_processing.py',
        'utils/hcc_mapping.py',
        'data/sample_data.py',
        'templates/index.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True

def test_python_syntax():
    """Test that Python files have valid syntax."""
    print("Testing Python syntax...")
    
    python_files = [
        'main.py',
        'models/risk_adjustment.py',
        'models/cost_drivers.py',
        'utils/data_processing.py',
        'utils/hcc_mapping.py',
        'data/sample_data.py'
    ]
    
    syntax_errors = []
    for file_path in python_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    compile(f.read(), file_path, 'exec')
                print(f"✓ {file_path} syntax OK")
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
                print(f"✗ {file_path} syntax error: {e}")
    
    if syntax_errors:
        print(f"✗ Syntax errors found: {syntax_errors}")
        return False
    else:
        print("✓ All Python files have valid syntax")
        return True

def test_imports():
    """Test that main modules can be imported (without dependencies)."""
    print("Testing imports...")
    
    try:
        # Test that we can at least read the files
        with open('main.py', 'r') as f:
            main_content = f.read()
        
        # Check for basic structure
        assert 'FastAPI' in main_content, "FastAPI not found in main.py"
        assert 'RiskAdjustmentModel' in main_content, "RiskAdjustmentModel not found in main.py"
        assert 'CostDriverAnalysis' in main_content, "CostDriverAnalysis not found in main.py"
        
        print("✓ Main module structure looks good")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_docker_config():
    """Test Docker configuration files."""
    print("Testing Docker configuration...")
    
    try:
        # Test Dockerfile
        with open('Dockerfile', 'r') as f:
            dockerfile_content = f.read()
        
        assert 'FROM python:3.11-slim' in dockerfile_content, "Wrong base image"
        assert 'COPY requirements.txt' in dockerfile_content, "Missing requirements copy"
        assert 'EXPOSE 8000' in dockerfile_content, "Missing port exposure"
        
        # Test docker-compose.yml
        with open('docker-compose.yml', 'r') as f:
            compose_content = f.read()
        
        assert 'version:' in compose_content, "Missing version in docker-compose"
        assert 'services:' in compose_content, "Missing services in docker-compose"
        assert 'claims-risk-adjustment:' in compose_content, "Missing main service"
        
        print("✓ Docker configuration looks good")
        return True
        
    except Exception as e:
        print(f"✗ Docker configuration test failed: {e}")
        return False

def test_requirements():
    """Test requirements.txt file."""
    print("Testing requirements...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        # Check for key dependencies
        key_deps = ['fastapi', 'pandas', 'numpy', 'scikit-learn', 'uvicorn']
        for dep in key_deps:
            assert dep in requirements.lower(), f"Missing dependency: {dep}"
        
        print("✓ Requirements file looks good")
        return True
        
    except Exception as e:
        print(f"✗ Requirements test failed: {e}")
        return False

def test_html_template():
    """Test HTML template."""
    print("Testing HTML template...")
    
    try:
        with open('templates/index.html', 'r') as f:
            html_content = f.read()
        
        # Check for basic HTML structure
        assert '<!DOCTYPE html>' in html_content, "Missing DOCTYPE"
        assert '<html' in html_content, "Missing html tag"
        assert '<head>' in html_content, "Missing head tag"
        assert '<body>' in html_content, "Missing body tag"
        assert 'Risk Adjustment' in html_content, "Missing title content"
        
        print("✓ HTML template looks good")
        return True
        
    except Exception as e:
        print(f"✗ HTML template test failed: {e}")
        return False

def test_gitignore():
    """Test .gitignore file."""
    print("Testing .gitignore...")
    
    try:
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()
        
        # Check for important patterns
        important_patterns = ['__pycache__', '*.pyc', '*.pkl', 'data/', 'models/']
        for pattern in important_patterns:
            assert pattern in gitignore_content, f"Missing pattern: {pattern}"
        
        print("✓ .gitignore looks good")
        return True
        
    except Exception as e:
        print(f"✗ .gitignore test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Claims-Based Risk Adjustment System - Basic Test Suite")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_python_syntax,
        test_imports,
        test_docker_config,
        test_requirements,
        test_html_template,
        test_gitignore
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The system structure is correct.")
        print("\nTo run the full system:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run with Docker: docker compose up --build")
        print("3. Or run directly: python main.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)