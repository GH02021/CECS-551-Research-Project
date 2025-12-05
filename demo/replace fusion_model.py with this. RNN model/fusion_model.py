
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
def data_create(csvfile: str):
    df = pd.read_csv(csvfile)

    target_col = df.columns[-1]
    ignore_cols = [df.columns[0], df.columns[1], target_col]

    X = df.drop(columns=ignore_cols).astype("float32")
    y = df[target_col].astype("float32").values.reshape(-1, 1)
    
    split_idx = int(len(df) * 0.7)
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]

    y_train = y[:split_idx]
    y_test  = y[split_idx:]

    return X_train, y_train, X_test, y_test
    '''


#RNN data
def data_create(csvfile: str, window: int = 5):
    df = pd.read_csv(csvfile)

    # 假设前两列是 Date, Ticker，最后一列是 label_up_next
    target_col = df.columns[-1]
    ignore_cols = [df.columns[0], df.columns[1], target_col]

    feature_cols = [c for c in df.columns if c not in ignore_cols]

    # 按 Ticker + Date 排序，保证时间顺序
    df = df.sort_values([df.columns[0], df.columns[1]])  # ["Date", "Ticker"] 也可以手写

    X_list, y_list = [], []

    # 按 ticker 分组，避免不同股票之间串窗
    for _, g in df.groupby(df.columns[0]):  # 按 Ticker 分组
        g = g.reset_index(drop=True)

        feats = g[feature_cols].astype("float32").values  # (n_days, F)
        labels = g[target_col].astype("float32").values   # (n_days,)

        if len(g) < window:
            continue  # 太短就跳过

        # 滑动窗口：每次取 window 天的序列
        for i in range(window - 1, len(g)):
            X_seq = feats[i-window+1 : i+1]   # (window, F)
            y = labels[i]                     # 用窗口最后一天的 label_up_next
            X_list.append(X_seq)
            y_list.append(y)

    X = np.stack(X_list).astype("float32")          # (N_samples, window, F)
    y = np.array(y_list, dtype="float32").reshape(-1, 1)

    # 按时间顺序切 7:3（此时样本已经是按时间排好的）
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test

#define the model
#MLP model
'''
class Fusionnet(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
'''


'''#CNN model
class Fusionnet(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden, kernel_size=3,padding=1),
            nn.ReLU(),
            #nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=hidden, out_channels=2*hidden, kernel_size=3,padding=1),
            nn.ReLU(),
            #nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=2*hidden, out_channels=4*hidden, kernel_size=3,padding=1),
            nn.ReLU(),
            #nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_dim)
            out = self.features(dummy)
            fc_in = out.numel()  

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(fc_in , 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)    
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)

        return x'''


#RNN model
class Fusionnet(nn.Module):
    def __init__(self, feature_dim, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=feature_dim,  # = F
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):           # x: (batch, T, F)
        out, (h_n, c_n) = self.rnn(x)
        h_last = h_n[-1]            # (batch, hidden)
        return self.classifier(h_last)


    

class FusionModel:

    def __init__(self, csvfile:str="Output/training_dateset.csv", hidden=64, lr=1e-3):
        self.device = device
        self.hidden = hidden
        self.lr = lr
        self.csvf = csvfile
        
        X_train, y_train, X_test, y_test = data_create(csvfile, window=5)

        N_train, T, F = X_train.shape
        N_test = X_test.shape[0]

        self.scaler = StandardScaler().fit(
            X_train.reshape(-1, F)          # (N_train * T, F)
        )

        X_train_scaled = self.scaler.transform(
            X_train.reshape(-1, F)
        ).reshape(N_train, T, F)

        X_test_scaled = self.scaler.transform(
            X_test.reshape(-1, F)
        ).reshape(N_test, T, F)
        
        self.Xt = torch.from_numpy(X_train_scaled.astype("float32")).to(self.device)
        self.Yt = torch.from_numpy(y_train.astype("float32")).to(self.device)

        self.Xv = torch.from_numpy(X_test_scaled.astype("float32")).to(self.device)
        self.Yv = torch.from_numpy(y_test.astype("float32")).to(self.device)

        _, T, F = X_train.shape
        feature_dim = F
        self.model = Fusionnet(feature_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    def train(self, num_epochs=50, patience=5):
        os.makedirs("Output", exist_ok=True)
        os.makedirs("Plot", exist_ok=True)

        best_val = float("inf")
        wait = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.Xt)
            loss = self.criterion(out, self.Yt)
            loss.backward()
            self.optimizer.step()

            
            log = f"Epoch {epoch+1:03d} | train_loss={loss.item():.4f}"

            # 简单用 test 当 val
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(self.Xv)
                val_loss = self.criterion(val_out, self.Yv).item()
            train_losses.append(loss.item())
            val_losses.append(val_loss)

            log += f", val_loss={val_loss:.4f}"
            #print(log)

            if patience is not None:
                if val_loss < best_val:
                    best_val = val_loss
                    wait = 0
                    torch.save(self.model.state_dict(), "Output/fusion_model.pth")
                else:
                    wait += 1
                    if wait >= patience:
                        #print("Early stopping.")
                        break

            plt.plot(train_losses, label="Train")
            plt.plot(val_losses, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Fusion MLP Learning Curve")
            plt.legend()
            plt.savefig("Plot/learning_curve.png", dpi=150)
            plt.close()



    def evaluate(self, fmp = "Output/fusion_model.pth"):
        if os.path.exists(fmp):
            self.model.load_state_dict(torch.load(fmp, map_location=self.device))
        self.model.eval()

        with torch.no_grad():
            probs = self.model(self.Xv).cpu().numpy().ravel()
        pred_labels = (probs >= 0.5).astype(int)
        true_labels = self.Yv.cpu().numpy().ravel().astype(int)

        acc  = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels, zero_division=0)
        rec  = recall_score(true_labels, pred_labels, zero_division=0)
        f1   = f1_score(true_labels, pred_labels, zero_division=0)

        metrics = {
            "accuracy":  float(acc),
            "precision": float(prec),
            "recall":    float(rec),
            "f1_score":  float(f1)
        }

        with open("Output/fusion.json", "w") as f:
            json.dump(metrics, f, indent=2)

        cm = confusion_matrix(true_labels, pred_labels)

        fig, ax = plt.subplots()
        ax.imshow(cm)

        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"])
        ax.set_yticklabels(["True 0","True 1"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        ax.set_title("Confusion Matrix - Fusion Model")
        fig.savefig("Plot/confusion_matrix.png", dpi=150)
        plt.close(fig)


    def predict(self, date:str, ticker:str, csvfile:str = None, fmp = "Output/fusion_model.pth"):
        
        if csvfile:
            self.csvf = csvfile 
        df = pd.read_csv(self.csvf)

        target_col = df.columns[-1]
        row = df[(df["Date"] == date) & (df["Ticker"] == ticker)]
        
        if row.empty:
            return None, None

        x = row.drop(columns=["Date", "Ticker", target_col]).astype("float32")
        x_scaled = self.scaler.transform(x)
        xt = torch.from_numpy(x_scaled.astype("float32")).to(self.device)
        
        if os.path.exists(fmp):
            self.model.load_state_dict(
                torch.load(fmp, map_location=self.device))

        self.model.eval()
        with torch.no_grad():
            p = self.model(xt).detach().cpu().numpy().ravel()[0]

        label = "UP" if p >= 0.5 else "DOWN"
        return label, float(p)

