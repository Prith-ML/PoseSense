# Live-Human-Action-Detection-Project

![skeleton gif](https://github.com/user-attachments/assets/a554513e-9a1c-4451-9cb2-93263b69591e)

The Live Human Action Detection Project is a computer vision application designed to recognize and classify human actions in real-time using only a webcam. It combines the power of pose estimation and deep learning to understand body movements and categorize them into predefined actions such as Clapping, Hand Waving, and Hopping.

The main goals of the project are:

1. To recognize human actions in a live video stream using 3D pose information.

2. To classify those actions using a temporal neural network (LSTM).

3. To provide a visual and interactive interface that shows real-time feedback to the user.

4. To explore pose-based action recognition without relying on raw RGB video or depth data.


                   ┌───────────────┐
                   │   Webcam      │
                   └──────┬────────┘
                          ↓
                 ┌─────────────────────┐
                 │ MediaPipe Pose (33) │
                 └──────┬──────────────┘
                          ↓
  convert_mediapipe_to_ntu25()  ← re-orders, averages ▶ 25 joints
                          ↓
        normalise_skeleton()   ← centre, rotate, scale
                          ↓
      Temporal queue (T=25)    ← sliding window
                          ↓
            LSTM (2×128)       ← PyTorch, many-to-one
                          ↓
         Softmax + label      → overlay text / bars on frame


![Skeletal joints video](https://github.com/user-attachments/assets/a2990d33-0f6c-4015-a325-75c5a9436a7f)



<img width="271" alt="skeletal visual" src="https://github.com/user-attachments/assets/8e92e860-7f38-4c0d-9313-58a4e2ef8975" />
