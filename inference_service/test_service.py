#!/usr/bin/env python3
"""
test_service.py
---------------
Quick test script to verify inference service is working.

Usage:
    python test_service.py
"""

import base64
import json
import requests
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def create_test_frame(size=(640, 480)):
    """Create a dummy test frame."""
    # Create random RGB image
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save temporarily
    test_path = Path('/tmp/test_frame.jpg')
    img.save(test_path, 'JPEG')
    
    return test_path


def image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_health_check(base_url='http://localhost:8000'):
    """Test /health endpoint."""
    print('\n1️⃣ Testing health check...')
    
    try:
        response = requests.get(f'{base_url}/health', timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print(f'   ✅ Status: {data["status"]}')
        print(f'   ✅ Model loaded: {data["model_loaded"]}')
        print(f'   ✅ Device: {data["device"]}')
        
        return True
    
    except requests.exceptions.ConnectionError:
        print('   ❌ Connection failed. Is the service running?')
        print('      Start with: docker-compose up -d')
        return False
    
    except Exception as e:
        print(f'   ❌ Health check failed: {e}')
        return False


def test_predict(base_url='http://localhost:8000'):
    """Test /predict endpoint."""
    print('\n2️⃣ Testing prediction...')
    
    try:
        # Create test frame
        test_frame_path = create_test_frame()
        frame_base64 = image_to_base64(test_frame_path)
        
        # Prepare request
        payload = {
            'frames': [frame_base64],
            'timestamp': '2024-01-15T10:30:00Z',
            'save_features': False
        }
        
        # Send request
        print('   📤 Sending request...')
        response = requests.post(
            f'{base_url}/predict',
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        print(f'   ✅ Anomaly Score: {result["anomaly_score"]:.3f}')
        print(f'   ✅ Is Anomaly: {result["is_anomaly"]}')
        print(f'   ✅ Predicted Class: {result["predicted_class"]}')
        print(f'   ✅ Confidence: {result["confidence"]:.3f}')
        print(f'   ✅ Processing Time: {result["processing_time_ms"]:.1f}ms')
        
        # Cleanup
        test_frame_path.unlink()
        
        return True
    
    except requests.exceptions.Timeout:
        print('   ❌ Request timeout. Model might be loading...')
        return False
    
    except Exception as e:
        print(f'   ❌ Prediction failed: {e}')
        return False


def test_api_documentation(base_url='http://localhost:8000'):
    """Test API docs endpoints."""
    print('\n3️⃣ Testing API documentation...')
    
    try:
        # Swagger UI
        response = requests.get(f'{base_url}/docs', timeout=5)
        if response.status_code == 200:
            print('   ✅ Swagger UI available at: {}/docs'.format(base_url))
        
        # ReDoc
        response = requests.get(f'{base_url}/redoc', timeout=5)
        if response.status_code == 200:
            print('   ✅ ReDoc available at: {}/redoc'.format(base_url))
        
        return True
    
    except Exception as e:
        print(f'   ⚠️  Documentation check failed: {e}')
        return False


def main():
    """Run all tests."""
    print('=' * 60)
    print('🧪 Anomaly Detection Service - Test Suite')
    print('=' * 60)
    
    base_url = 'http://localhost:8000'
    
    # Run tests
    health_ok = test_health_check(base_url)
    
    if not health_ok:
        print('\n❌ Service is not healthy. Aborting tests.')
        sys.exit(1)
    
    predict_ok = test_predict(base_url)
    docs_ok = test_api_documentation(base_url)
    
    # Summary
    print('\n' + '=' * 60)
    print('📊 Test Summary')
    print('=' * 60)
    print(f'Health Check: {"✅ PASS" if health_ok else "❌ FAIL"}')
    print(f'Prediction:   {"✅ PASS" if predict_ok else "❌ FAIL"}')
    print(f'API Docs:     {"✅ PASS" if docs_ok else "❌ FAIL"}')
    print('=' * 60)
    
    if all([health_ok, predict_ok, docs_ok]):
        print('\n🎉 All tests passed! Service is ready for production.')
        sys.exit(0)
    else:
        print('\n⚠️  Some tests failed. Check logs: docker-compose logs')
        sys.exit(1)


if __name__ == '__main__':
    main()