from __future__ import annotations

import numpy as np


class MetricAggregator:

    def __init__(self):

        self.reset()

    def update(self, predictions, targets):

        self.predictions.append(predictions)

        self.targets.append(targets)

    def reset(self):

        self.predictions = []

        self.targets = []

    def compute(self):

        predictions = np.concatenate(self.predictions)

        targets = np.concatenate(self.targets)

        return predictions, targets