# Live-Human-Action-Detection-Project

![skeleton gif](https://github.com/user-attachments/assets/a554513e-9a1c-4451-9cb2-93263b69591e)

The Live Human Action Detection Project is a computer vision application designed to recognize and classify human actions in real-time using only a webcam. It combines the power of pose estimation and deep learning to understand body movements and categorize them into predefined actions such as Clapping, Hand Waving, and Hopping.

The main goals of the project are:

1. To recognize human actions in a live video stream using 3D pose information.

2. To classify those actions using a temporal neural network (LSTM).

3. To provide a visual and interactive interface that shows real-time feedback to the user.

4. To explore pose-based action recognition without relying on raw RGB video or depth data.


# How the system works 

Instead of analyzing the raw video feed, the system uses a real-time pose estimation engine (e.g., MediaPipe, OpenPose) to extract 3D joint keypoints from each frame. Each person's pose is converted into a set of vectors — for instance, the x, y, z positions of the shoulders, elbows, knees, etc. This provides a simplified but highly informative representation of body position and posture.
The key benefit is that this is much lighter than video analysis. You're working with maybe 33 joints per frame instead of hundreds of thousands of pixels.

Human actions are dynamic — they unfold over time. So, rather than classifying a single frame, this system builds a temporal window of pose frames, typically spanning around 30–60 frames (1–2 seconds of motion). This sequence of pose data becomes the input to the neural network.

The core of the model is an LSTM (Long Short-Term Memory) network. LSTMs are a type of recurrent neural network (RNN) designed for learning from sequences — they're particularly well-suited for recognizing patterns that depend on time, like human gestures or actions.
In this project, the LSTM takes in the sequence of joint coordinates and outputs a prediction: a label representing the recognized action. For example, based on how the joints move over a few seconds, it might output "clapping" or "hopping."

# Visual Inference 

<img src="https://github.com/user-attachments/assets/a2990d33-0f6c-4015-a325-75c5a9436a7f" width="50%"/>

This is a video from a live 3D animation of human pose data, rendered using the NTU RGB+D 25-joint skeleton format.
Each green dot in the image is a 3D point corresponding to a joint in the body (like the wrist, elbow, or shoulder), and the yellow lines represent bones — that is, the anatomical connections between those joints.

When a pose sequence is passed to the LSTM, each time step processes one frame's vector, updating the hidden state of the network. As the sequence unfolds — wrists moving inward, then pausing at the center, then retracting — the LSTM learns to associate this pattern with the "clapping" label. It recognizes not just positions, but the trajectory and timing of joint movements.








<img width="500" alt="skeletal visual" src="https://github.com/user-attachments/assets/8e92e860-7f38-4c0d-9313-58a4e2ef8975" />

This visualization represents a centered and aligned skeleton frame, a crucial preprocessing step in pose-based deep learning. Here, the skeleton has been translated so that the hip joint is at the origin (0, 0, 0), and the coordinate axes are reoriented to follow a canonical frame: the X-axis aligns with the shoulders, the Y-axis follows the spine vertically, and the Z-axis points forward in depth. This normalization is done to remove variations caused by the subject's position, orientation, or camera angle, ensuring that identical actions (like clapping or waving) result in consistent joint trajectories regardless of how or where the action is performed. By standardizing the pose data in this way, the LSTM model can focus purely on the motion pattern itself, rather than being confused by irrelevant spatial differences. 

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (built-in or USB)
- At least 4GB RAM (8GB recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/PoseSense.git
   cd PoseSense
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test your system**
   ```bash
   python test_system.py
   ```

4. **Run the demo**
   ```bash
   python run_demo.py
   ```

## 🎯 Features

### Real-Time Action Recognition
- **Live webcam processing** with minimal latency
- **3D pose estimation** using MediaPipe
- **Temporal analysis** with LSTM neural network
- **Instant feedback** with confidence scores

### Supported Actions
- **Clapping** - Hands moving together in front of chest
- **Hand Waving** - Arm moving side to side
- **Hopping** - Up and down jumping movement

### Professional Visualization
- **Color-coded skeleton** with different colors for body parts
- **Joint classification** (central, limb, extremity)
- **Real-time metrics** (FPS, buffer status, confidence)
- **Interactive UI** with semi-transparent overlays

### Performance Features
- **GPU acceleration** support (CUDA)
- **Configurable settings** for different hardware
- **Efficient processing** (25 joints vs. full video frames)
- **Optimized inference** pipeline

## 🛠️ Usage

### Basic Operation
1. **Start the application** using `python run_demo.py`
2. **Position yourself** in front of the webcam
3. **Perform actions** like clapping, waving, or hopping
4. **Watch real-time results** with skeleton visualization
5. **Press 'q'** to quit the application

### Testing Different Actions
- **Clapping**: Bring hands together in front of chest
- **Hand Waving**: Move one arm side to side
- **Hopping**: Jump up and down in place

### Tips for Best Results
- Ensure **good lighting** for better pose detection
- Stay **centered in frame** for optimal tracking
- Use **clear, deliberate movements**
- Maintain **steady camera position**

## 📁 Project Structure

```
PoseSense/
├── 📁 src/                    # Source code
│   ├── 📁 core/              # Main application logic
│   ├── 📁 utils/             # Utility scripts
│   └── 📁 models/            # Pre-trained models
├── 📁 tests/                 # Testing scripts
├── 📁 examples/              # Usage examples
├── main.py                   # Main entry point
├── README.md                 # Project overview
├── PROJECT_STRUCTURE.md      # Detailed structure
├── CONTRIBUTING.md           # Contribution guide
├── LICENSE                   # MIT License
├── Dataset                   # Dataset instructions
├── requirements.txt          # Dependencies
└── setup.py                  # Package setup
```

## 🔧 Configuration

The system can be customized through `config.py`:

- **Model settings**: Input size, hidden layers, sequence length
- **Pose estimation**: Detection confidence, model complexity
- **Visualization**: Colors, joint sizes, UI elements
- **Performance**: GPU settings, batch size, thresholds

## 🧪 Testing & Troubleshooting

### System Test
Run comprehensive testing to verify all components:
```bash
python tests/test_system.py
```

### Common Issues
1. **Camera not working**: Check permissions and drivers
2. **Dependencies fail**: Use `pip install -r requirements.txt`
3. **Model file missing**: Ensure `pytorchModel.pth` is present
4. **Performance issues**: Lower MediaPipe complexity in config

### Performance Optimization
- **GPU users**: Ensure CUDA is properly installed
- **CPU users**: Lower MediaPipe model complexity
- **Memory issues**: Reduce sequence length in config

## 📊 Technical Details

### Model Architecture
- **Input**: 25 frames × 25 joints × 3 coordinates = 75 features
- **LSTM**: 2 layers with 128 hidden units
- **Output**: 3-class classification with confidence scores
- **Training**: Adam optimizer with CrossEntropy loss

### Data Processing
- **Pose extraction**: MediaPipe 3D landmarks
- **Format conversion**: NTU RGB+D 25-joint skeleton
- **Preprocessing**: Centering, alignment, scale normalization
- **Temporal window**: 25-frame sequence for action recognition

### Performance Metrics
- **Real-time processing**: 30+ FPS on modern hardware
- **Latency**: <50ms inference time
- **Accuracy**: High precision for trained actions
- **Memory usage**: Efficient skeleton representation

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- **New action classes** and training data
- **Performance optimization** and GPU acceleration
- **UI improvements** and visualization enhancements
- **Documentation** and tutorial creation

## 📚 Related Work

This project builds upon:
- **MediaPipe Pose**: Real-time pose estimation
- **NTU RGB+D Dataset**: Skeleton format and training data
- **LSTM Networks**: Temporal sequence modeling
- **Computer Vision**: Real-time video processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe team** for pose estimation technology
- **NTU RGB+D dataset** contributors for training data
- **PyTorch community** for deep learning framework
- **OpenCV contributors** for computer vision tools

## 📞 Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Wiki**: Detailed tutorials and troubleshooting

---

**Star this repository if you find it useful! ⭐**

*Built with ❤️ using Python, PyTorch, and MediaPipe* 
