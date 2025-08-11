"""
Configuration file for PoseSense
Modify these settings to customize the system behavior.
"""

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'src/models/pytorchModel.pth',
    'input_size': 75,  # 25 joints × 3 coordinates
    'hidden_size': 128,
    'num_layers': 2,
    'num_classes': 3,
    'sequence_length': 25,  # Number of frames to process
}

# Pose Estimation Configuration
POSE_CONFIG = {
    'model_complexity': 2,  # 0, 1, or 2 (higher = more accurate but slower)
    'smooth_landmarks': True,
    'enable_segmentation': False,
    'smooth_segmentation': False,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'show_skeleton': True,
    'show_ui_panel': True,
    'show_legend': True,
    'skeleton_thickness': 3,
    'joint_sizes': {
        'central': 8,
        'limb': 6,
        'extremity': 4,
    },
    'colors': {
        'body': (255, 51, 153),      # Pink
        'left_arm': (0, 255, 255),   # Yellow
        'right_arm': (0, 165, 255),  # Orange
        'left_leg': (153, 0, 255),   # Purple
        'right_leg': (255, 0, 102),  # Red-Pink
        'central_joint': (0, 0, 255),    # Red
        'limb_joint': (0, 255, 0),       # Green
        'extremity_joint': (255, 0, 0),  # Blue
    }
}

# Action Labels
ACTION_LABELS = ['Clapping', 'Hand Waving', 'Hopping']
LABEL_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red

# Camera Configuration
CAMERA_CONFIG = {
    'camera_index': 0,  # Change to 1, 2, etc. for different cameras
    'frame_width': 640,
    'frame_height': 480,
    'fps_target': 30,
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'enable_gpu': True,  # Set to False if you have GPU issues
    'batch_size': 1,
    'confidence_threshold': 0.3,  # Minimum confidence to show prediction
} 