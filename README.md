# Live-Human-Action-Detection-Project

![skeleton gif](https://github.com/user-attachments/assets/a554513e-9a1c-4451-9cb2-93263b69591e)

The Live Human Action Detection Project is a computer vision application designed to recognize and classify human actions in real-time using only a webcam. It combines the power of pose estimation and deep learning to understand body movements and categorize them into predefined actions such as Clapping, Hand Waving, and Hopping.

The main goals of the project are:

1. To recognize human actions in a live video stream using 3D pose information.

2. To classify those actions using a temporal neural network (LSTM).

3. To provide a visual and interactive interface that shows real-time feedback to the user.

4. To explore pose-based action recognition without relying on raw RGB video or depth data.


# How the system works 

Instead of analyzing the raw video feed, the system uses a real-time pose estimation engine (e.g., MediaPipe, OpenPose) to extract 3D joint keypoints from each frame. Each person’s pose is converted into a set of vectors — for instance, the x, y, z positions of the shoulders, elbows, knees, etc. This provides a simplified but highly informative representation of body position and posture.
The key benefit is that this is much lighter than video analysis. You’re working with maybe 33 joints per frame instead of hundreds of thousands of pixels.

Human actions are dynamic — they unfold over time. So, rather than classifying a single frame, this system builds a temporal window of pose frames, typically spanning around 30–60 frames (1–2 seconds of motion). This sequence of pose data becomes the input to the neural network.

The core of the model is an LSTM (Long Short-Term Memory) network. LSTMs are a type of recurrent neural network (RNN) designed for learning from sequences — they’re particularly well-suited for recognizing patterns that depend on time, like human gestures or actions.
In this project, the LSTM takes in the sequence of joint coordinates and outputs a prediction: a label representing the recognized action. For example, based on how the joints move over a few seconds, it might output “clapping” or “hopping.”

# Visual Inference 

<img src="https://github.com/user-attachments/assets/a2990d33-0f6c-4015-a325-75c5a9436a7f" width="50%"/>

This is a video from a live 3D animation of human pose data, rendered using the NTU RGB+D 25-joint skeleton format.
Each green dot in the image is a 3D point corresponding to a joint in the body (like the wrist, elbow, or shoulder), and the yellow lines represent bones — that is, the anatomical connections between those joints.

When a pose sequence is passed to the LSTM, each time step processes one frame's vector, updating the hidden state of the network. As the sequence unfolds — wrists moving inward, then pausing at the center, then retracting — the LSTM learns to associate this pattern with the "clapping" label. It recognizes not just positions, but the trajectory and timing of joint movements.






<img width="500" alt="skeletal visual" src="https://github.com/user-attachments/assets/8e92e860-7f38-4c0d-9313-58a4e2ef8975" />

This visualization represents a centered and aligned skeleton frame, a crucial preprocessing step in pose-based deep learning. Here, the skeleton has been translated so that the hip joint is at the origin (0, 0, 0), and the coordinate axes are reoriented to follow a canonical frame: the X-axis aligns with the shoulders, the Y-axis follows the spine vertically, and the Z-axis points forward in depth. This normalization is done to remove variations caused by the subject’s position, orientation, or camera angle, ensuring that identical actions (like clapping or waving) result in consistent joint trajectories regardless of how or where the action is performed. By standardizing the pose data in this way, the LSTM model can focus purely on the motion pattern itself, rather than being confused by irrelevant spatial differences. 
