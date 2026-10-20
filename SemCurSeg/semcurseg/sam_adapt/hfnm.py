"""High-Frequency Noise Masking (HFNM): wavelet-domain noise augmentation for SAM Adapt training."""
import cv2
import numpy as np
import pywt


class HFNMTransform:
    """Adds Gaussian noise to the high-frequency DWT sub-bands (LH, HL, HH) of each colour channel."""

    def __init__(self, wavelet='haar', noise_std=0.1, noise_prob=0.5):
        self.wavelet = wavelet
        self.noise_std = noise_std
        self.noise_prob = noise_prob

    def __call__(self, image, mask=None):
        """image: [H, W, 3] uint8 or float in [0, 1]; the mask is returned unchanged."""
        if np.random.rand() > self.noise_prob or self.noise_std <= 0:
            return image, mask
        is_uint8 = image.dtype == np.uint8
        img = image.astype(np.float32) / 255.0 if is_uint8 else image.astype(np.float32)
        channels = []
        for c in range(img.shape[2]):
            ll, (lh, hl, hh) = pywt.dwt2(img[:, :, c], self.wavelet)
            noisy = (lh + np.random.randn(*lh.shape) * self.noise_std,
                     hl + np.random.randn(*hl.shape) * self.noise_std,
                     hh + np.random.randn(*hh.shape) * self.noise_std)
            rec = pywt.idwt2((ll, noisy), self.wavelet)
            if rec.shape != img.shape[:2]:
                rec = cv2.resize(rec, (img.shape[1], img.shape[0]))
            channels.append(rec)
        out = np.clip(np.stack(channels, axis=-1), 0, 1)
        return ((out * 255).astype(np.uint8) if is_uint8 else out), mask
