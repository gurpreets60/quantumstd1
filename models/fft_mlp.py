"""
FFT + MLP — frequency-domain features fed into a neural network.

Amplitude QNN analog: quantum amplitude encoding naturally operates in
frequency/amplitude space. FFT magnitude extracts global frequency patterns
without positional information (translation-invariant to some extent).

rfft2 on a 28x28 image → complex (28, 15) spectrum → magnitude (28, 15)
→ flattened (420,) feature vector for MLP.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from .base import BaseModel


class FFTMLPModel(BaseModel):
    def __init__(self):
        super().__init__("FFT_MLP")
        self.model = MLPClassifier(
            hidden_layer_sizes=(512, 256),  # two hidden layers
            max_iter=50,
            random_state=42,
        )

    def _fft_features(self, X):
        """Compute 2D FFT magnitude on each 28x28 image.

        rfft2 exploits Hermitian symmetry → output shape (28, 15) per image.
        Result: (N, 420) float32 array of frequency magnitudes.
        """
        imgs = X.reshape(-1, 28, 28)                    # unflatten: (N, 28, 28)
        mag  = np.abs(np.fft.rfft2(imgs))               # FFT magnitude: (N, 28, 15)
        return mag.reshape(len(X), -1).astype(np.float32)  # flatten: (N, 420)

    def train(self, X, y):
        self.model.fit(self._fft_features(X), y)

    def predict_proba(self, X):
        return self.model.predict_proba(self._fft_features(X)).astype(np.float32)
