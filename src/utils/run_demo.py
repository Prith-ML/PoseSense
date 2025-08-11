#!/usr/bin/env python3
"""
Quick Start Demo for PoseSense
Run this script to start the live action detection system.
"""

import sys
import os

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = ['cv2', 'mediapipe', 'torch', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mediapipe':
                import mediapipe
            elif package == 'torch':
                import torch
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Install them using:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_model_file():
    """Check if the model file exists."""
    model_file = "pytorchModel.pth"
    if not os.path.exists(model_file):
        print(f"❌ Model file '{model_file}' not found!")
        print("Please download the pre-trained model or train your own.")
        return False
    
    print(f"✅ Model file '{model_file}' found!")
    return True

def main():
    """Main function to run the demo."""
    print("🚀 PoseSense - Live Human Action Detection")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        sys.exit(1)
    
    print("\n🎯 Starting live action detection...")
    print("📹 Make sure your webcam is connected and accessible.")
    print("⌨️  Press 'q' to quit the application.")
    print("=" * 50)
    
    try:
        # Import and run the main application
        from src.core.liveApplicationCode import *
        print("✅ Application started successfully!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure your webcam is working")
        print("2. Check if the model file is compatible")
        print("3. Try running 'python main.py' directly")
        sys.exit(1)

if __name__ == "__main__":
    main() 