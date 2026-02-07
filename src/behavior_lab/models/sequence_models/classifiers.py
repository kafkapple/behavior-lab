"""Sequence-based action classifiers: RuleBased, MLP, LSTM, Transformer."""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

from behavior_lab.core.types import ClassificationResult, ModelMetrics

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BaseActionClassifier(ABC):
    """Abstract base class for action classifiers with fit/predict/evaluate interface."""

    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict: pass

    @abstractmethod
    def predict(self, X) -> ClassificationResult: pass

    def evaluate(self, X_test, y_test) -> ModelMetrics:
        result = self.predict(X_test)
        y_true = y_test.flatten()
        y_pred = result.predictions.flatten()
        accuracy = float(np.mean(y_true == y_pred))

        f1_per_class = {}
        f1_scores = []
        for i, name in enumerate(self.class_names):
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_per_class[name] = f1
            f1_scores.append(f1)

        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                cm[int(t), int(p)] += 1

        return ModelMetrics(
            accuracy=accuracy, f1_macro=float(np.mean(f1_scores)),
            per_class_metrics=f1_per_class, confusion_matrix=cm)


class RuleBasedClassifier(BaseActionClassifier):
    """Velocity threshold baseline (no training required)."""

    def __init__(self, num_classes=4, class_names=None, fps=30.0, thresholds=None):
        super().__init__(num_classes, class_names or ["stationary", "walking", "running", "other"])
        self.fps = fps
        self.thresholds = thresholds or {"stationary": 0.5, "walking": 3.0}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return {"status": "no_training_required"}

    def predict(self, X):
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)
        N, T, F = X.shape
        positions = X[:, :, :2]
        velocity = np.zeros((N, T))
        for i in range(N):
            disp = np.diff(positions[i], axis=0)
            vel = np.linalg.norm(disp, axis=1)
            velocity[i, 1:] = vel
            velocity[i, 0] = vel[0] if len(vel) > 0 else 0

        predictions = np.zeros((N, T), dtype=np.int32)
        probabilities = np.zeros((N, T, self.num_classes))
        for i in range(N):
            for j in range(T):
                v = velocity[i, j]
                if v < self.thresholds["stationary"]:
                    predictions[i, j] = 0
                    probabilities[i, j] = [0.8, 0.15, 0.03, 0.02]
                elif v < self.thresholds["walking"]:
                    predictions[i, j] = 1
                    probabilities[i, j] = [0.1, 0.7, 0.15, 0.05]
                else:
                    predictions[i, j] = 2
                    probabilities[i, j] = [0.05, 0.15, 0.75, 0.05]

        return ClassificationResult(
            predictions=predictions.flatten(),
            probabilities=probabilities.reshape(-1, self.num_classes),
            class_names=self.class_names)


class MLPClassifier(BaseActionClassifier):
    """Frame-by-frame MLP with optional temporal windowing."""

    def __init__(self, num_classes=4, class_names=None, input_dim=14,
                 hidden_dims=None, window_size=1, learning_rate=0.001,
                 epochs=50, batch_size=64):
        super().__init__(num_classes, class_names)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MLPClassifier")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.window_size = window_size
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        in_f = input_dim * window_size
        for hd in self.hidden_dims:
            layers.extend([nn.Linear(in_f, hd), nn.ReLU(), nn.Dropout(0.3)])
            in_f = hd
        layers.append(nn.Linear(in_f, num_classes))
        self.model = nn.Sequential(*layers).to(self.device)

    def _create_windows(self, X, y=None):
        if X.ndim == 2: X = X.reshape(1, *X.shape)
        N, T, F = X.shape
        hw = self.window_size // 2
        X_w, y_w = [], []
        for i in range(N):
            for j in range(T):
                s, e = max(0, j - hw), min(T, j + hw + 1)
                w = X[i, s:e]
                if w.shape[0] < self.window_size:
                    pb = max(0, hw - j)
                    pa = self.window_size - w.shape[0] - pb
                    w = np.pad(w, ((pb, pa), (0, 0)), mode='edge')
                X_w.append(w.flatten())
                if y is not None:
                    y_w.append(y[i, j] if y.ndim > 1 else y[j])
        return np.array(X_w), np.array(y_w) if y is not None else None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xw, yw = self._create_windows(X_train, y_train)
        ds = TensorDataset(torch.FloatTensor(Xw), torch.LongTensor(yw))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        for ep in range(self.epochs):
            self.model.train()
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                crit(self.model(xb), yb).backward()
                opt.step()
        return {}

    def predict(self, X):
        Xw, _ = self._create_windows(X)
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(Xw).to(self.device))
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
        return ClassificationResult(predictions=preds, probabilities=probs,
                                    class_names=self.class_names)


class LSTMClassifier(BaseActionClassifier):
    """Bidirectional LSTM for sequence classification."""

    def __init__(self, num_classes=4, class_names=None, input_dim=14,
                 hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.3,
                 learning_rate=0.001, epochs=50, batch_size=32, seq_len=64):
        super().__init__(num_classes, class_names)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTMClassifier")
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _LSTM(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                                 bidirectional=bidirectional,
                                 dropout=dropout if num_layers > 1 else 0)
                out_dim = hidden_dim * 2 if bidirectional else hidden_dim
                s.fc = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(out_dim, hidden_dim),
                    nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
            def forward(s, x):
                return s.fc(s.lstm(x)[0])

        self.model = _LSTM().to(self.device)
        self.lr = learning_rate

    def _prepare_sequences(self, X, y=None):
        if X.ndim == 2: X = X.reshape(1, *X.shape)
        if y is not None and y.ndim == 1: y = y.reshape(1, -1)
        Xs, ys = [], []
        for i in range(len(X)):
            for s in range(0, len(X[i]) - self.seq_len + 1, self.seq_len // 2):
                Xs.append(X[i, s:s + self.seq_len])
                if y is not None: ys.append(y[i, s:s + self.seq_len])
        return np.array(Xs) if Xs else np.empty((0,)), np.array(ys) if ys else None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xs, ys = self._prepare_sequences(X_train, y_train)
        if len(Xs) == 0: return {}
        ds = TensorDataset(torch.FloatTensor(Xs), torch.LongTensor(ys))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        for ep in range(self.epochs):
            self.model.train()
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                out = self.model(xb).view(-1, self.num_classes)
                crit(out, yb.view(-1)).backward()
                opt.step()
        return {}

    def predict(self, X):
        if X.ndim == 2: X = X.reshape(1, *X.shape)
        self.model.eval()
        all_preds, all_probs = [], []
        for seq in X:
            T = len(seq)
            preds = np.zeros(T, dtype=np.int32)
            probs = np.zeros((T, self.num_classes))
            with torch.no_grad():
                for s in range(0, T, self.seq_len):
                    e = min(s + self.seq_len, T)
                    chunk = seq[s:e]
                    if len(chunk) < self.seq_len:
                        chunk = np.pad(chunk, ((0, self.seq_len - len(chunk)), (0, 0)), mode='edge')
                    out = self.model(torch.FloatTensor(chunk).unsqueeze(0).to(self.device))
                    p = F.softmax(out, dim=2).squeeze(0).cpu().numpy()
                    preds[s:e] = out.argmax(dim=2).squeeze(0).cpu().numpy()[:e - s]
                    probs[s:e] = p[:e - s]
            all_preds.extend(preds)
            all_probs.extend(probs)
        return ClassificationResult(predictions=np.array(all_preds),
                                    probabilities=np.array(all_probs),
                                    class_names=self.class_names)


class TransformerClassifier(BaseActionClassifier):
    """Transformer encoder for sequence classification."""

    def __init__(self, num_classes=4, class_names=None, input_dim=14,
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=256,
                 dropout=0.1, learning_rate=0.0001, epochs=50, batch_size=32, seq_len=64):
        super().__init__(num_classes, class_names)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Transformer(nn.Module):
            def __init__(s):
                super().__init__()
                s.input_proj = nn.Linear(input_dim, d_model)
                pe = torch.zeros(5000, d_model)
                pos = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
                div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                s.register_buffer('pe', pe.unsqueeze(0))
                layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
                s.transformer = nn.TransformerEncoder(layer, num_layers)
                s.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, d_model // 2),
                                     nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes))
            def forward(s, x):
                x = s.input_proj(x) + s.pe[:, :x.size(1)]
                return s.fc(s.transformer(x))

        self.model = _Transformer().to(self.device)
        self.lr = learning_rate

    def _prepare_sequences(self, X, y=None):
        if X.ndim == 2: X = X.reshape(1, *X.shape)
        if y is not None and y.ndim == 1: y = y.reshape(1, -1)
        Xs, ys = [], []
        for i in range(len(X)):
            for s in range(0, len(X[i]) - self.seq_len + 1, self.seq_len // 2):
                Xs.append(X[i, s:s + self.seq_len])
                if y is not None: ys.append(y[i, s:s + self.seq_len])
        return np.array(Xs) if Xs else np.empty((0,)), np.array(ys) if ys else None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xs, ys = self._prepare_sequences(X_train, y_train)
        if len(Xs) == 0: return {}
        ds = TensorDataset(torch.FloatTensor(Xs), torch.LongTensor(ys))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.CrossEntropyLoss()
        for ep in range(self.epochs):
            self.model.train()
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                out = self.model(xb).view(-1, self.num_classes)
                crit(out, yb.view(-1)).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            sched.step()
        return {}

    def predict(self, X):
        if X.ndim == 2: X = X.reshape(1, *X.shape)
        self.model.eval()
        all_preds, all_probs = [], []
        for seq in X:
            T = len(seq)
            preds = np.zeros(T, dtype=np.int32)
            probs = np.zeros((T, self.num_classes))
            with torch.no_grad():
                for s in range(0, T, self.seq_len):
                    e = min(s + self.seq_len, T)
                    chunk = seq[s:e]
                    if len(chunk) < self.seq_len:
                        chunk = np.pad(chunk, ((0, self.seq_len - len(chunk)), (0, 0)), mode='edge')
                    out = self.model(torch.FloatTensor(chunk).unsqueeze(0).to(self.device))
                    p = F.softmax(out, dim=2).squeeze(0).cpu().numpy()
                    preds[s:e] = out.argmax(dim=2).squeeze(0).cpu().numpy()[:e - s]
                    probs[s:e] = p[:e - s]
            all_preds.extend(preds)
            all_probs.extend(probs)
        return ClassificationResult(predictions=np.array(all_preds),
                                    probabilities=np.array(all_probs),
                                    class_names=self.class_names)


def get_action_classifier(model_name: str, num_classes: int = 4,
                          class_names: List[str] = None, **kwargs) -> BaseActionClassifier:
    """Factory: get classifier by name."""
    name = model_name.lower().replace("-", "_")
    if name in ("rule_based", "rule", "baseline"):
        return RuleBasedClassifier(num_classes, class_names, **kwargs)
    elif name == "mlp":
        return MLPClassifier(num_classes, class_names, **kwargs)
    elif name == "lstm":
        return LSTMClassifier(num_classes, class_names, **kwargs)
    elif name in ("transformer", "attention"):
        return TransformerClassifier(num_classes, class_names, **kwargs)
    raise ValueError(f"Unknown model: {model_name}. Available: rule_based, mlp, lstm, transformer")
