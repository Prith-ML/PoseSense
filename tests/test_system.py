#!/usr/bin/env python3
"""
System Test Script for PoseSense
Run this to test all components and diagnose issues.
"""

import sys
import time
import traceback
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    packages = {
        'OpenCV': 'cv2',
        'MediaPipe': 'mediapipe',
        'PyTorch': 'torch',
        'NumPy': 'numpy',
        'Scikit-learn': 'sklearn',
    }
    
    failed_imports = []
    
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError as e:
            print(f"  ❌ {package_name}: {e}")
            failed_imports.append(package_name)
    
    return len(failed_imports) == 0

def test_gpu():
    """Test GPU availability for PyTorch."""
    print("\n🔍 Testing GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ GPU available: {gpu_name}")
            print(f"  📊 GPU count: {gpu_count}")
            return True
        else:
            print("  ⚠️  GPU not available, using CPU")
            return False
    except Exception as e:
        print(f"  ❌ Error checking GPU: {e}")
        return False

def test_model_file():
    """Test if the model file exists and can be loaded."""
    print("\n🔍 Testing model file...")
    
    model_path = Path("src/models/pytorchModel.pth")
    if not model_path.exists():
        print(f"  ❌ Model file not found: {model_path}")
        return False
    
    file_size = model_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  ✅ Model file found: {model_path}")
    print(f"  📊 File size: {file_size:.1f} MB")
    
    # Try to load the model
    try:
        import torch
        from src.core.liveApplicationCode import SkeletonLSTM
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SkeletonLSTM()
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device)
        model.eval()
        print("  ✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return False

def test_camera():
    """Test camera access."""
    print("\n🔍 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  ❌ Cannot open camera")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("  ❌ Cannot read from camera")
            cap.release()
            return False
        
        height, width = frame.shape[:2]
        print(f"  ✅ Camera working: {width}x{height}")
        cap.release()
        return True
    except Exception as e:
        print(f"  ❌ Camera error: {e}")
        return False

def test_pose_estimation():
    """Test MediaPipe pose estimation."""
    print("\n🔍 Testing pose estimation...")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pose.process(dummy_image)
        
        print("  ✅ Pose estimation initialized")
        pose.close()
        return True
    except Exception as e:
        print(f"  ❌ Pose estimation error: {e}")
        return False

def run_performance_test():
    """Run a quick performance test."""
    print("\n🔍 Running performance test...")
    
    try:
        import torch
        from src.core.liveApplicationCode import SkeletonLSTM
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SkeletonLSTM()
        model.load_state_dict(torch.load("src/models/pytorchModel.pth", map_location=device))
        model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 25, 25, 3).to(device)
        
        # Warm up
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Performance test
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"  ✅ Performance test completed")
        print(f"  📊 Average inference time: {avg_time:.2f} ms")
        print(f"  📊 Theoretical FPS: {fps:.1f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Performance test error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 PoseSense System Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("GPU Availability", test_gpu),
        ("Model File", test_model_file),
        ("Camera Access", test_camera),
        ("Pose Estimation", test_pose_estimation),
        ("Performance", run_performance_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready to run PoseSense.")
        print("\n🚀 To start the application, run:")
        print("   python main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Common solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check camera permissions")
        print("3. Ensure model file is in the correct location")
        print("4. Check GPU drivers if using CUDA")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 