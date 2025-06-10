import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

data_folder = "filterWWW/"
action_map = {'010': 0, '023': 1, '026': 2}

# --- Axis Normalization ---
def rotate_skeleton_upright(skeleton):
    if skeleton.shape != (25, 3):
        return skeleton
    try:
        hip = skeleton[0]
        neck = skeleton[20]
        r_shoulder = skeleton[4]
        l_shoulder = skeleton[8]

        y_axis = neck - hip
        y_axis /= np.linalg.norm(y_axis) + 1e-8

        x_axis = l_shoulder - r_shoulder
        x_axis /= np.linalg.norm(x_axis) + 1e-8

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-8

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-8

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        return skeleton @ R
    except:
        return skeleton

# --- Scale Normalization ---
def normalize_skeleton_scale(skeleton):
    hip = skeleton[0]
    neck = skeleton[20]
    spine_len = np.linalg.norm(neck - hip) + 1e-8
    return skeleton / spine_len

# --- Parser ---
def parse_skeleton_file_centered(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    try:
        frames = int(lines[i].strip())
    except ValueError:
        return None
    i += 1

    sequence = []

    for _ in range(frames):
        while i < len(lines):
            try:
                num_bodies = int(lines[i].strip())
                i += 1
                break
            except ValueError:
                i += 1
        if i >= len(lines):
            break

        frame_data = np.zeros((25, 3))

        for b in range(num_bodies):
            i += 1
            if i >= len(lines): break
            try:
                num_joints = int(lines[i].strip())
            except ValueError:
                break
            i += 1

            joints = []
            for j in range(num_joints):
                if i >= len(lines): break
                joint_info = list(map(float, lines[i].strip().split()))
                joints.append(joint_info[:3])
                i += 1

            frame_data = np.array(joints)
            break

        if len(frame_data) == 25:
            frame_data = frame_data - frame_data[0:1]
            frame_data = rotate_skeleton_upright(frame_data)
            frame_data = normalize_skeleton_scale(frame_data)
            sequence.append(frame_data)

    return np.array(sequence)

# Load all data
X, Y = [], []
for fname in tqdm(os.listdir(data_folder)):
    if fname.endswith(".skeleton") and 'A' in fname:
        label = action_map.get(fname.split('A')[1][:3], -1)
        if label == -1:
            continue
        path = os.path.join(data_folder, fname)
        sequence = parse_skeleton_file_centered(path)
        if sequence is not None and sequence.shape[0] > 0:
            X.append(sequence)
            Y.append(label)

# Pad sequences to length 64
max_len = 64
X_padded = []
for seq in X:
    if seq.shape[0] < max_len:
        pad = np.zeros((max_len - seq.shape[0], 25, 3))
        seq = np.concatenate([seq, pad])
    else:
        seq = seq[:max_len]
    X_padded.append(seq)

X_padded = np.stack(X_padded)
Y = np.array(Y)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_padded, Y, test_size=0.2, stratify=Y, random_state=42
)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

# Dataset class
class SkeletonDataset(Dataset):
    def __init__(self, X_path, Y_path):
        self.data = np.load(X_path)
        self.labels = np.load(Y_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# LSTM model
class SkeletonLSTM(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=2, num_classes=3):
        super(SkeletonLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, t, j, c = x.shape
        x = x.view(b, t, j * c)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkeletonLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(SkeletonDataset("X_train.npy", "Y_train.npy"), batch_size=32, shuffle=True)
test_loader = DataLoader(SkeletonDataset("X_test.npy", "Y_test.npy"), batch_size=32)

for epoch in range(100):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out, 1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)
        loss_sum += loss.item()

    acc = 100 * correct / total
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            _, pred = torch.max(out, 1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/100 | Loss: {loss_sum:.4f} | Train Acc: {acc:.2f}% | Test Acc: {val_acc:.2f}%")
    torch.save(model.state_dict(), "NEWNEWwithnewactionsAlignedqskeleton_lstm.pth")
