"""Scikit-learn compatible temporal regressor with serializable inference payload."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .pipeline import _sequence_schema, _deep_arrays


class TemporalEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, kind, features, seed=42, hidden=32, learning_rate=0.003,
                 weight_decay=0.0001, batch_size=32, epochs=10, cancelled=None, progress=None):
        self.kind, self.features, self.seed = kind, features, seed
        self.hidden, self.learning_rate, self.weight_decay = hidden, learning_rate, weight_decay
        self.batch_size, self.epochs = batch_size, epochs
        self.cancelled, self.progress = cancelled, progress

    def fit(self, matrix, target):
        import torch
        from .deep import TemporalRegressor
        from .tuning import TrainingCancelled
        self.schema_ = _sequence_schema(self.features)
        if self.kind == "CNN-Transformer" and len(self.schema_["variables"]) < 2:
            raise ValueError("CNN-Transformer needs at least two temporal variables")
        torch.manual_seed(self.seed)
        # CPU deterministic profile is intentionally the same across CV folds.
        sequence, static = _deep_arrays(matrix, self.features, self.schema_)
        self.config_ = {"sequence_channels": sequence.shape[2], "static_features": static.shape[1],
                        "timesteps": sequence.shape[1], "hidden": self.hidden}
        self.model_ = TemporalRegressor(self.kind, **self.config_)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(sequence), torch.from_numpy(static),
                                                  torch.from_numpy(np.asarray(target, dtype=np.float32)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True,
                                             generator=torch.Generator().manual_seed(self.seed))
        self.loss_history_ = []
        for epoch in range(self.epochs):
            if self.cancelled and self.cancelled():
                raise TrainingCancelled("Cancelled at epoch boundary")
            self.model_.train()
            losses = []
            for seq, sta, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(self.model_(seq, sta), y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.loss_history_.append(float(np.mean(losses)))
            if self.progress:
                self.progress({"stage": f"epoch {epoch + 1}/{self.epochs}; MSE={self.loss_history_[-1]:.5f}",
                               "percent": 25 + round(55 * (epoch + 1) / self.epochs)})
        return self

    def predict(self, matrix):
        import torch
        sequence, static = _deep_arrays(matrix, self.features, self.schema_)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy(sequence), torch.from_numpy(static)).numpy().astype(float)

    def payload(self):
        return {"model_config": self.config_, "sequence_schema": self.schema_,
                "state_dict": {k: v.detach().cpu() for k, v in self.model_.state_dict().items()}}
