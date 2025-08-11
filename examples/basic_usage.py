#!/usr/bin/env python3
"""
Basic Usage Example for PoseSense

This example shows how to use the PoseSense system programmatically
for pose estimation and action recognition.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def example_pose_estimation():
    """Example of using pose estimation without the full application."""
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        from src.core.config import POSE_CONFIG
        
        print("🎯 PoseSense Basic Usage Example")
        print("=" * 40)
        
        # Initialize MediaPipe pose estimation
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=POSE_CONFIG['model_complexity'],
            min_detection_confidence=POSE_CONFIG['min_detection_confidence'],
            min_tracking_confidence=POSE_CONFIG['min_tracking_confidence']
        )
        
        print("✅ MediaPipe pose estimation initialized")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("📹 Webcam opened successfully")
        print("⌨️  Press 'q' to quit, 's' to save a frame")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame for pose detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            # Draw pose landmarks if detected
            if results.pose_landmarks:
                # Convert to drawing utils
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Get landmark coordinates
                landmarks = results.pose_landmarks.landmark
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"📍 Frame {frame_count}: {len(landmarks)} landmarks detected")
            
            # Display frame
            cv2.imshow("PoseSense Basic Example", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"pose_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved as {filename}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("✅ Example completed successfully")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Make sure you have all dependencies installed:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

def example_model_loading():
    """Example of loading the trained model."""
    try:
        import torch
        from src.core.liveApplicationCode import SkeletonLSTM
        from src.core.config import MODEL_CONFIG
        
        print("\n🧠 Model Loading Example")
        print("=" * 30)
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {device}")
        
        # Create model
        model = SkeletonLSTM(
            input_size=MODEL_CONFIG['input_size'],
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_layers=MODEL_CONFIG['num_layers'],
            num_classes=MODEL_CONFIG['num_classes']
        )
        
        # Load trained weights
        model_path = MODEL_CONFIG['model_path']
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print("✅ Model loaded successfully")
            
            # Model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📊 Total parameters: {total_params:,}")
            print(f"📊 Model architecture: {MODEL_CONFIG['num_layers']} LSTM layers")
            print(f"📊 Hidden size: {MODEL_CONFIG['hidden_size']}")
        else:
            print(f"❌ Model file not found: {model_path}")
            print("💡 Download the pre-trained model or train your own")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def main():
    """Main example function."""
    print("🚀 PoseSense Examples")
    print("=" * 50)
    
    # Run examples
    example_pose_estimation()
    example_model_loading()
    
    print("\n🎉 All examples completed!")
    print("💡 Check the saved images to see pose detection in action")

if __name__ == "__main__":
    main() 