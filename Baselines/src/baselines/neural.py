"""MLP nuisance backend shared by front-door and proximal estimators."""
from __future__ import annotations

import copy
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn

CONFIG = dict(hidden_dimensions=[128, 128], activation='ReLU', optimizer='Adam',
              learning_rate=1e-3, weight_decay=1e-4, max_epochs=300, batch_size=128,
              patience=30, min_epochs=20, validation_fraction=0.2, cross_fitting_folds=0,
              refit_all_context_after_epoch_selection=True)


def network(width, output=1):
    return nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 128),
                         nn.ReLU(), nn.Linear(128, output))


def setup(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def optimize(model, x, y, *, seed, classification=False, validation=None, epochs=300):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
    rng = np.random.default_rng(seed)
    best, state, best_epoch, stale = float('inf'), None, 0, 0
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            ix = order[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[ix]).reshape(-1), y[ix])
            loss.backward()
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                score = float(criterion(model(validation[0]).reshape(-1), validation[1]).item())
            if score < best - 1e-6:
                best, best_epoch, stale = score, epoch, 0
                state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= 30 and epoch >= 20:
                break
    if validation is not None:
        model.load_state_dict(state)
    else:
        best_epoch = epochs
    model.eval()
    return dict(selected_epochs=best_epoch, validation_loss=best if validation is not None else None,
                epochs_executed=epoch)


def standardize(train, other):
    mean, sd = train.mean(0), np.maximum(train.std(0), 1e-6)
    return (train-mean)/sd, (other-mean)/sd, mean, sd


class MLPNuisance:
    def __init__(self, seed, classification, log):
        self.seed, self.classification, self.log = seed, classification, log

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        device = setup(self.seed)
        train, val = train_test_split(np.arange(len(x)), test_size=.2, random_state=self.seed,
                                     stratify=y if self.classification else None)
        xt, xv, _, _ = standardize(x[train], x[val])
        ym = 0. if self.classification else y[train].mean()
        ys = 1. if self.classification else max(y[train].std(), 1e-6)
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        model = network(x.shape[1]).to(device)
        info = optimize(model, tensor(xt), tensor((y[train]-ym)/ys), seed=self.seed,
                        classification=self.classification,
                        validation=(tensor(xv), tensor((y[val]-ym)/ys)))
        setup(self.seed)
        self.xm, self.xs = x.mean(0), np.maximum(x.std(0), 1e-6)
        self.ym = 0. if self.classification else y.mean()
        self.ys = 1. if self.classification else max(y.std(), 1e-6)
        self.model = network(x.shape[1]).to(device)
        optimize(self.model, tensor((x-self.xm)/self.xs), tensor((y-self.ym)/self.ys),
                 seed=self.seed, classification=self.classification, epochs=info['selected_epochs'])
        self.device = device
        self.log.append(dict(seed=self.seed, binary=self.classification, n_fit=len(x),
                             n_validation=len(val), **info))
        return self

    def _raw(self, x):
        tensor = torch.as_tensor((np.asarray(x)-self.xm)/self.xs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(tensor).reshape(-1)

    def predict(self, x):
        return self._raw(x).cpu().numpy()*self.ys + self.ym

    def predict_proba(self, x):
        p = torch.sigmoid(self._raw(x)).cpu().numpy()
        return np.column_stack([1-p, p])
