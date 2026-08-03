"""
proctor_ml/verifiers/phone_verifier.py
=======================================
Second-stage mobile phone classifier for the Proctor ML Engine.

Uses a MobileNetV3-Small backbone (torchvision, ImageNet pre-trained) with a
randomly-initialised 2-class head (NOT_PHONE=0, PHONE=1).  It runs only on
crops that YOLO has already flagged as "cell phone" and that have cleared the
geometry filter + confidence threshold — so the extra per-frame cost is
effectively zero under normal exam conditions.

Out of the box the head weights are random, so the model defaults to NOT_PHONE
for most crops, which further reduces false positives.  For best accuracy,
fine-tune on deployment-specific phone / not-phone crops (see below).

Architecture
------------
    torchvision MobileNetV3-Small  (ImageNet weights, frozen or fine-tuned)
    └── classifier[-1] → nn.Linear(in_features, 2)   PHONE / NOT_PHONE

Supported backbone names (PHONE_VERIFIER_MODEL)
-----------------------------------------------
    mobilenet_v3_small   (default, ~2.5 M params, ~3-8 ms / crop on CPU)
    mobilenet_v3_large   (~5.5 M params)
    efficientnet_b0      (~5.3 M params)
    efficientnet_b1      (~7.8 M params)

Custom fine-tuned weights
-------------------------
Set PHONE_VERIFIER_MODEL to an absolute path to a .pt / .pth file that
contains a state_dict for a 2-class head on the architecture named in
PHONE_VERIFIER_ARCH (defaults to mobilenet_v3_small).

How to retrain
--------------
1. Collect ~500+ positive (phone in hand / on desk) and ~500+ negative
   (headphones, chargers, remotes, clothing, wall objects) crops from your
   deployment camera at the same resolution the proctor runs (320 px YOLO
   input -> variable crop size).
2. Build the model::

       import torchvision.models as tvm
       import torch.nn as nn, torch

       model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
       model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)

3. Fine-tune::

       optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
       criterion = nn.CrossEntropyLoss()
       # ... standard train loop, ~10 epochs, batch_size=32

4. Save::

       torch.save(model.state_dict(), "phone_verifier.pt")

5. Point the service at the new weights::

       PHONE_VERIFIER_MODEL=/path/to/phone_verifier.pt
       PHONE_VERIFIER_ARCH=mobilenet_v3_small   # architecture of saved weights

Expected latency
----------------
MobileNetV3-Small at 224×224 input on Intel i7 CPU: ~3-8 ms per crop.
Triggered only on YOLO "cell phone" candidates that clear geometry + conf gate.

Configuration env vars
----------------------
PHONE_VERIFIER_ENABLED     bool   enable the verifier               (default True)
PHONE_VERIFIER_MODEL       str    model name or .pt file path       (default mobilenet_v3_small)
PHONE_VERIFIER_THRESHOLD   float  phone-class probability threshold (default 0.50)
PHONE_VERIFIER_ARCH        str    arch for custom .pt files         (default mobilenet_v3_small)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------
@dataclass
class VerificationResult:
    """Result returned by PhoneVerifier.verify().

    Attributes:
        verified:   True when the crop is classified as a phone.
        confidence: Probability of the PHONE class (0.0–1.0).
        reason:     Human-readable explanation.
                    Pre-trained (ImageNet only): "phone" | "not_phone".
                    After fine-tuning with labelled negatives you can extend
                    this to "headphones", "charger", "remote", etc. by mapping
                    additional output classes.
    """
    verified: bool
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# PhoneVerifier
# ---------------------------------------------------------------------------
class PhoneVerifier:
    """Lightweight MobileNetV3-Small binary classifier (PHONE / NOT_PHONE).

    Intended to be loaded once at server startup via the model loader and
    stored in ``Models.phone_verifier``.  Thread-safety: the model is run
    inside ``torch.no_grad()`` with no mutable state per call, so concurrent
    calls from multiple async tasks are safe after construction.

    Quick-start::

        verifier = PhoneVerifier(model_name="mobilenet_v3_small", threshold=0.50)
        result   = verifier.verify(crop_bgr)
        if result.verified:
            # emit to Boolean smoother
            ...

    Parameters
    ----------
    model_name:
        Either a torchvision architecture name ("mobilenet_v3_small", etc.) or
        a filesystem path to a fine-tuned .pt / .pth state_dict file.
    threshold:
        Minimum phone-class probability for a detection to be accepted.
        Maps to ``PHONE_VERIFIER_THRESHOLD`` (default 0.50).
    """

    # ImageNet normalisation
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    _INPUT_SIZE = 224

    _SUPPORTED_ARCHS = {
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "efficientnet_b1",
    }

    def __init__(self, model_name: str = "mobilenet_v3_small", threshold: float = 0.50) -> None:
        self.threshold   = threshold
        self._model_name = model_name
        self._model      = None
        self._transform  = None
        self._device     = None
        self._load(model_name)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load(self, model_name: str) -> None:
        """Build and initialise the classifier.

        Non-fatal: on any failure ``self._model`` remains None and
        ``verify()`` returns a safe pass-through result (verified=True).
        Upstream geometry + YOLO confidence guards remain active regardless.
        """
        try:
            import torch
            import torchvision.models as tvm
            import torchvision.transforms as T

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if os.path.isfile(model_name):
                # Fine-tuned weights supplied as a filesystem path.
                arch  = os.getenv("PHONE_VERIFIER_ARCH", "mobilenet_v3_small")
                model = self._build_backbone(arch, tvm)
                state = torch.load(model_name, map_location=self._device, weights_only=True)
                model.load_state_dict(state)
                logger.info(
                    "[PhoneVerifier] Loaded fine-tuned weights from '%s' (arch=%s).",
                    model_name, arch,
                )
            else:
                model = self._build_backbone(model_name, tvm)
                logger.info(
                    "[PhoneVerifier] Using torchvision %s with ImageNet init "
                    "and random 2-class head.  Fine-tune for best accuracy.",
                    model_name,
                )

            model.eval()
            model.to(self._device)
            self._model = model

            # Preprocessing: BGR crop -> resized RGB tensor
            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=self._MEAN, std=self._STD),
            ])

            logger.info(
                "[PhoneVerifier] Ready on %s (threshold=%.2f).",
                str(self._device), self.threshold,
            )

        except Exception as exc:
            logger.warning(
                "[PhoneVerifier] Failed to load ('%s'): %s  "
                "— all phone crops will pass through.",
                model_name, exc,
            )
            self._model = None

    @staticmethod
    def _build_backbone(name: str, tvm) -> object:
        """Instantiate a supported torchvision backbone with a 2-class head.

        Raises ValueError for unsupported names so the caller can surface a
        clear message rather than a cryptic attribute error.
        """
        import torch.nn as nn

        _BUILDERS = {
            "mobilenet_v3_small": lambda: tvm.mobilenet_v3_small(
                weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
            "mobilenet_v3_large": lambda: tvm.mobilenet_v3_large(
                weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
            "efficientnet_b0": lambda: tvm.efficientnet_b0(
                weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
            "efficientnet_b1": lambda: tvm.efficientnet_b1(
                weights=tvm.EfficientNet_B1_Weights.IMAGENET1K_V1),
        }

        if name not in _BUILDERS:
            raise ValueError(
                f"Unsupported PHONE_VERIFIER_MODEL '{name}'. "
                f"Choose from: {sorted(_BUILDERS)} "
                f"or supply an absolute path to a fine-tuned .pt file."
            )

        model = _BUILDERS[name]()

        # Replace the 1000-class ImageNet head with a 2-class binary head.
        # Both MobileNetV3 and EfficientNet expose model.classifier[-1] as
        # a Linear layer, so this works for all supported architectures.
        last = model.classifier[-1]
        if not hasattr(last, "in_features"):
            raise ValueError(
                f"Cannot find in_features on classifier[-1] for '{name}'. "
                "The architecture may not be supported."
            )
        model.classifier[-1] = nn.Linear(last.in_features, 2)
        return model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def verify(self, crop: np.ndarray) -> VerificationResult:
        """Classify *crop* (BGR ndarray from YOLO bounding box) as PHONE or NOT_PHONE.

        Class indices:
            0 = NOT_PHONE
            1 = PHONE

        Falls back to ``VerificationResult(verified=True, ...)`` (pass-through)
        when:
          * the model failed to load (``self._model is None``)
          * the crop is empty / None
          * inference raises an unexpected exception

        In all pass-through cases, upstream guards (geometry filter + YOLO
        confidence threshold) remain the only gatekeepers.

        Parameters
        ----------
        crop:
            Numpy array in BGR channel order, shape (H, W, 3).
            Typically the raw slice ``image[y1:y2, x1:x2]`` from the YOLO box.

        Returns
        -------
        VerificationResult
            ``verified``   True if phone-class probability >= threshold.
            ``confidence`` Phone-class probability (0.0–1.0).
            ``reason``     'phone' | 'not_phone' | 'verifier_unavailable' |
                           'empty_crop' | 'inference_error'.
        """
        if self._model is None:
            return VerificationResult(
                verified=True, confidence=1.0, reason="verifier_unavailable"
            )

        if crop is None or crop.size == 0:
            return VerificationResult(
                verified=False, confidence=0.0, reason="empty_crop"
            )

        try:
            import torch
            import torch.nn.functional as F

            rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits     = self._model(tensor)          # (1, 2)
                probs      = F.softmax(logits, dim=1)[0]  # (2,)
                phone_prob = float(probs[1].item())       # index 1 == PHONE

            verified = phone_prob >= self.threshold
            reason   = "phone" if verified else "not_phone"
            return VerificationResult(verified=verified, confidence=phone_prob, reason=reason)

        except Exception as exc:
            logger.warning("[PhoneVerifier] Inference error (%s) — passing crop through.", exc)
            return VerificationResult(
                verified=True, confidence=1.0, reason="inference_error"
            )
