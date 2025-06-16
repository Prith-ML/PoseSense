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

![Skeletal joints video](https://github.com/user-attachments/assets/a2990d33-0f6c-4015-a325-75c5a9436a7f)



<img width="271" alt="skeletal visual" src="https://github.com/user-attachments/assets/8e92e860-7f38-4c0d-9313-58a4e2ef8975" />
