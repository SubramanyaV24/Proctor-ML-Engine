"""
L2CS-Net gaze estimation wrapper.

L2CS-Net (Abdelrahman et al., arXiv:2203.03339) is a two-branch ResNet-50 CNN
that regresses gaze pitch and yaw from the full face crop.  Because it operates
on the whole face image rather than on isolated pupil/iris landmarks, it is
significantly more robust to eyeglass glare than geometry-based methods.

Performance:
  - 3.92 deg mean angular error on MPIIGaze
  - 10.41 deg mean angular error on Gaze360

This module provides a lightweight wrapper around the l2cs package that:
  - Pre-processes the face crop (resize to 448x448, ImageNet normalise)
  - Runs inference using the loaded ResNet-50 model
  - Decodes the 90-bin softmax output into pitch/yaw angles (degrees)
  - Returns None gracefully when the model is not available

Reference: https://github.com/Ahmednull/L2CS-Net
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Lazy imports — the module is imported only when L2CS is available so the
# service starts fine without the l2cs package installed.
_l2cs_available: Optional[bool] = None


def _check_l2cs() -> bool:
    global _l2cs_available
    if _l2cs_available is not None:
        return _l2cs_available
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        from l2cs import Pipeline  # noqa: F401
        _l2cs_available = True
    except ImportError:
        _l2cs_available = False
    return _l2cs_available


class L2CSGazeModel:
    """Wraps the L2CS-Net ResNet-50 model for gaze angle regression.

    Instead of using the high-level l2cs.Pipeline (which runs its own face
    detector internally), we load only the gaze model and run inference on
    pre-cropped face images provided by the Proctor pipeline's own face
    detector.  This avoids running a redundant second face detection pass.
    """

    # ImageNet normalisation constants
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _IMAGE_SIZE = 448
    # Gaze angle bins: 90 bins spanning -90 to +90 degrees
    _IDX_TENSOR = np.arange(90, dtype=np.float32)

    def __init__(self, checkpoint_path: str, device: object) -> None:
        """Load the L2CS-Net model from *checkpoint_path*.

        Args:
            checkpoint_path: Absolute path to L2CSNet_gaze360.pkl
            device:          torch.device (cpu or cuda)
        """
        import torch
        import torch.nn as nn
        from l2cs import Pipeline  # used only to borrow the model class

        self.device = device

        # L2CS-Net ships its own model definition inside the l2cs package.
        # We instantiate it directly to control the input (pre-cropped face).
        # The Pipeline class loads the same ResNet-50 architecture.
        softmax = nn.Softmax(dim=1)

        # Load the full Pipeline to extract the trained model weights cleanly
        # (l2cs.Pipeline handles the arch init + checkpoint loading correctly)
        _pipeline = Pipeline(
            weights=checkpoint_path,
            arch="ResNet50",
            device=device,
        )
        # Extract the underlying torch model from the pipeline
        self._model = _pipeline.model
        self._model.eval()
        self._softmax = softmax
        self._device  = device

    # ------------------------------------------------------------------
    def predict(self, face_crop: np.ndarray) -> Optional[Tuple[float, float]]:
        """Estimate gaze from a face crop (BGR uint8).

        Returns (pitch_deg, yaw_deg) or None if inference fails.
        Pitch > 0 means looking up; Yaw > 0 means looking right.
        """
        try:
            import torch
            tensor = self._preprocess(face_crop)
            with torch.no_grad():
                pitch_out, yaw_out = self._model(tensor)
            pitch_deg = self._decode(pitch_out)
            yaw_deg   = self._decode(yaw_out)
            return float(pitch_deg), float(yaw_deg)
        except Exception as exc:
            print(f"[L2CS] Inference error: {exc}")
            return None

    # ------------------------------------------------------------------
    def _preprocess(self, face_crop: np.ndarray) -> "torch.Tensor":
        """Resize to 448x448, convert BGR→RGB, normalise, add batch dim."""
        import cv2
        import torch

        img = cv2.resize(face_crop, (self._IMAGE_SIZE, self._IMAGE_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
        img = (img - self._MEAN) / self._STD
        img = img.transpose(2, 0, 1)                       # HWC→CHW
        tensor = torch.from_numpy(img).unsqueeze(0).to(self._device)
        return tensor

    # ------------------------------------------------------------------
    def _decode(self, logits: "torch.Tensor") -> float:
        """Decode 90-bin softmax into a gaze angle (degrees).

        L2CS uses a 90-bin classification + regression head spanning -45..+45 deg
        (Gaze360 model; each bin = 1 deg). The final angle is the soft-max
        weighted expectation of bin centres.
        """
        import torch
        probs = self._softmax(logits).cpu().numpy()[0]            # (90,)
        idx   = self._IDX_TENSOR                                  # 0..89
        angle = (probs * idx).sum() * 4 - 180                    # map to -180..+180
        # Gaze360 model convention: bins span the full sphere but frontal gaze
        # concentrates around 0. Clamp to a reasonable proctoring range.
        return float(np.clip(angle, -90.0, 90.0))
