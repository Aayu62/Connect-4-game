"""
train_model.py  —  Train the ValueNet
===============================================
Run from your project root:
    pip install torch
    python train_model.py

What it does:
  - Loads data/connect4_dataset.npz
  - Trains a neural network to predict board value from Player 1's perspective
  - Output: +1 (winning position), -1 (losing), 0 (neutral)
  - Saves trained weights to data/value_net.pt

Architecture:
  Input (42)  →  Linear(128) → ReLU → Dropout
              →  Linear(64)  → ReLU → Dropout
              →  Linear(1)   → Tanh
              =  score in [-1, +1]
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import os
import time

#Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#Hyperparameters
BATCH_SIZE    = 256
EPOCHS        = 40
LR            = 1e-3
WEIGHT_DECAY  = 1e-4   # L2 regularization to prevent overfitting
DROPOUT       = 0.3
VAL_SPLIT     = 0.15   # 15% held out for validation
SAVE_PATH     = "data/value_net.pt"


#Model Definition
class ValueNet(nn.Module):
    """
    Predicts how good a board position is for Player 1.

    Input:  42 floats  (+1 = P1 piece, -1 = P2 piece, 0 = empty)
    Output: 1 float in [-1, +1]
              +1 → P1 is winning
               0 → neutral
              -1 → P1 is losing

    Dropout layers prevent the network from memorizing specific games
    and force it to learn general patterns (2-in-a-row threats etc.)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(64, 1),
            nn.Tanh()          # squashes output to [-1, +1]
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # shape: (batch,)

    def predict(self, board_np):
        """
        Convenience method: takes a raw numpy board (6×7),
        returns a single float score.
        Used by minimax in Stage 4.
        """
        encoded = encode_board(board_np)
        tensor  = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            return self.forward(tensor).item()


#Board encoding (same as generate_data.py)
def encode_board(board):
    encoded = np.zeros(42, dtype=np.float32)
    for r in range(6):
        for c in range(7):
            idx = r * 7 + c
            if board[r][c] == 1:
                encoded[idx] =  1.0
            elif board[r][c] == 2:
                encoded[idx] = -1.0
    return encoded


#Data Loading
def load_data(path="data/connect4_dataset.npz"):
    data = np.load(path)
    X = data['X']   # (N, 42)  float32
    y = data['y']   # (N,)     float32  values: -1, 0, +1
    print(f"\n  Loaded {len(X):,} positions from {path}")
    print(f"  Label distribution:  "
          f"+1={int((y==1).sum()):,}  "
          f"-1={int((y==-1).sum()):,}  "
          f" 0={int((y==0).sum()):,}")
    return X, y


def make_dataloaders(X, y):
    """
    Split into train/val, handle class imbalance with WeightedRandomSampler.
    Our dataset has ~2x more +1 labels than -1 — without balancing,
    the model would just learn to predict +1 all the time.
    """
    N = len(X)
    val_size   = int(N * VAL_SPLIT)
    train_size = N - val_size

    # Random split
    indices = np.random.permutation(N)
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

    # Weighted sampler: give equal chance to +1, -1, 0 samples
    label_map  = {1.0: 0, -1.0: 1, 0.0: 2}
    class_counts = np.array([
        (y_train == 1.0).sum(),
        (y_train == -1.0).sum(),
        (y_train == 0.0).sum()
    ], dtype=float)
    class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid div/0
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label_map[float(lbl)]] for lbl in y_train])
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float32),
        num_samples = len(sample_weights),
        replacement = True
    )

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t,   y_val_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n  Train: {train_size:,}  |  Val: {val_size:,}")
    return train_loader, val_loader


#Training Loop
def train(model, train_loader, val_loader):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Reduce LR if val loss plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss  = float('inf')
    best_model_state = None
    history = {'train': [], 'val': []}

    print(f"\n  Device: {device}")
    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>10}  {'LR':>10}")
    print(f"  {'─'*5}  {'─'*11}  {'─'*10}  {'─'*10}")

    for epoch in range(1, EPOCHS + 1):
        #Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        #Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds    = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  ← best"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5}  {train_loss:>11.5f}  {val_loss:>10.5f}  "
                  f"{current_lr:>10.6f}{marker}")

    # Restore best weights
    model.load_state_dict(best_model_state)
    print(f"\n  ✅ Best val loss: {best_val_loss:.5f}")
    return model, history


#Evaluation
def evaluate(model, val_loader):
    """
    Check prediction direction accuracy:
    Does the model correctly predict WIN vs LOSS vs DRAW direction?
    (Not exact value — just whether it scores winning positions higher.)
    """
    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)

            # Directional accuracy: sign should match
            pred_sign  = torch.sign(preds)
            label_sign = torch.sign(y_batch)
            correct   += (pred_sign == label_sign).sum().item()
            total     += len(y_batch)

    acc = 100.0 * correct / total
    print(f"\n  Directional accuracy on val set: {acc:.1f}%")
    print(f"  (How often the model correctly predicts win/loss/draw direction)")
    return acc


#Main 
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  Stage 3: Training ValueNet")
    print(f"{'='*55}")

    # 1. Load data
    X, y = load_data()

    # 2. Build dataloaders
    train_loader, val_loader = make_dataloaders(X, y)

    # 3. Build model
    model = ValueNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {total_params:,}")
    print(f"  Architecture: 42 → 128 → 64 → 1 (Tanh)\n")

    # 4. Train
    t0 = time.time()
    model, history = train(model, train_loader, val_loader)
    print(f"  Training time: {time.time() - t0:.1f}s")

    # 5. Evaluate
    evaluate(model, val_loader)

    # 6. Save
    os.makedirs("data", exist_ok=True)
    torch.save({
        'model_state_dict' : model.state_dict(),
        'architecture'     : '42→128→64→1',
        'val_loss'         : history['val'][-1],
        'epochs_trained'   : len(history['train']),
    }, SAVE_PATH)
    print(f"\n  Model saved to {SAVE_PATH}")
    print(f"\n{'='*55}\n")