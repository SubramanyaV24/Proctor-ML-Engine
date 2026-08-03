"""
SCRFD detector wrapper — InsightFace SCRFD ONNX runtime implementation.

Handles pre-processing (letterbox resize), ONNX inference, and post-processing
(NMS + landmark extraction) to produce the standard face dict format:
    {x, y, w, h, score, landmarks: {right_eye, left_eye, nose, right_mouth, left_mouth}}

Reference: InsightFace SCRFD (arXiv:2105.04714, ICLR 2022)
  WIDER FACE Hard AP: SCRFD-10GF=83.05%, SCRFD-2.5GF=77.87%

IMPORTANT: InsightFace pretrained ONNX weights are licensed for non-commercial
research use only. Only enable SCRFD_ENABLED=1 if your deployment qualifies.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class SCRFDDetector:
    """Thin wrapper around an onnxruntime InferenceSession for SCRFD models.

    The SCRFD ONNX model (bnkps variant) outputs bounding boxes and 5 facial
    landmark keypoints per detection.  This class handles:
      - Letterbox resizing to the model's fixed input size
      - Running ONNX inference
      - Decoding raw anchor-based detections
      - Per-class NMS
      - Coordinate remapping back to original image space
    """

    # ImageNet-standard normalisation (used by SCRFD preprocessing)
    _MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
    _STD  = np.array([128.0, 128.0, 128.0], dtype=np.float32)

    def __init__(
        self,
        session: object,          # onnxruntime.InferenceSession
        input_size: int = 640,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        self.session = session
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        # Resolve input/output names once
        self._input_name: str = session.get_inputs()[0].name
        self._output_names: List[str] = [o.name for o in session.get_outputs()]

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[dict]:
        """Run SCRFD on *image* (BGR uint8). Returns list of face dicts."""
        ih, iw = image.shape[:2]
        blob, scale, (pad_x, pad_y) = self._preprocess(image)

        outputs = self.session.run(
            self._output_names,
            {self._input_name: blob},
        )

        faces = self._postprocess(outputs, iw, ih, scale, pad_x, pad_y)
        return faces

    # ------------------------------------------------------------------
    def _preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize + normalise.  Returns (blob, scale, (pad_x, pad_y))."""
        ih, iw = image.shape[:2]
        sz = self.input_size
        scale = min(sz / iw, sz / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        pad_x = (sz - nw) // 2
        pad_y = (sz - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        blob = canvas.astype(np.float32)
        blob = (blob - self._MEAN) / self._STD
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW
        return blob, scale, (pad_x, pad_y)

    # ------------------------------------------------------------------
    def _postprocess(
        self,
        outputs: List[np.ndarray],
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> List[dict]:
        """Decode SCRFD outputs → face dicts in original image coordinates."""
        named = {n: arr for n, arr in zip(self._output_names, outputs)}

        score_keys = sorted(k for k in named if "score" in k)
        bbox_keys  = sorted(k for k in named if "bbox"  in k)
        kps_keys   = sorted(k for k in named if "kps"   in k)

        if not score_keys:
            n = len(outputs) // 3
            score_keys = self._output_names[:n]
            bbox_keys  = self._output_names[n:2*n]
            kps_keys   = self._output_names[2*n:]

        all_boxes = []
        all_scores = []
        all_kps = []

        strides = [8, 16, 32]
        for stride, sk, bk, kk in zip(strides, score_keys, bbox_keys, kps_keys):
            scores = named[sk].reshape(-1)
            bboxes = named[bk].reshape(-1, 4) * stride
            kps_arr = named.get(kk)
            if kps_arr is not None:
                kps_arr = kps_arr.reshape(-1, 10) * stride
            
            # Number of locations
            fh = self.input_size // stride
            fw = self.input_size // stride
            num_anchors = scores.shape[0] // (fh * fw)
            
            # Generate anchor centers for this stride
            anchor_centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers]*num_anchors, axis=1).reshape((-1, 2))
            
            keep = scores >= self.score_threshold
            scores = scores[keep]
            bboxes = bboxes[keep]
            anchor_centers = anchor_centers[keep]

            if len(scores) == 0:
                continue

            # bboxes output is distance to l, t, r, b from anchor center
            x1 = anchor_centers[:, 0] - bboxes[:, 0]
            y1 = anchor_centers[:, 1] - bboxes[:, 1]
            x2 = anchor_centers[:, 0] + bboxes[:, 2]
            y2 = anchor_centers[:, 1] + bboxes[:, 3]

            # Remap from letterboxed input coords back to original image coords
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
            all_scores.append(scores)

            if kps_arr is not None and kk in named:
                kp = kps_arr[keep]
                kps_x = kp[:, 0::2] + anchor_centers[:, 0:1]
                kps_y = kp[:, 1::2] + anchor_centers[:, 1:2]
                
                kps_x = (kps_x - pad_x) / scale
                kps_y = (kps_y - pad_y) / scale
                
                kp_decoded = np.zeros((len(scores), 5, 2), dtype=np.float32)
                kp_decoded[:, :, 0] = kps_x
                kp_decoded[:, :, 1] = kps_y
                all_kps.append(kp_decoded)
            else:
                all_kps.append(np.zeros((len(scores), 5, 2)))

        if not all_boxes:
            return []

        boxes  = np.concatenate(all_boxes,  axis=0)
        scores = np.concatenate(all_scores, axis=0)
        kps    = np.concatenate(all_kps,    axis=0)

        # OpenCV NMS (expects x,y,w,h format)
        wh_boxes = np.stack([
            boxes[:, 0], boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            wh_boxes,
            scores.tolist(),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )
        if len(indices) == 0:
            return []
        indices = indices.flatten()

        faces: List[dict] = []
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(orig_w, int(x2)); y2 = min(orig_h, int(y2))
            w  = x2 - x1; h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            # SCRFD 5-landmark order: right_eye, left_eye, nose, right_mouth, left_mouth
            kp = kps[i]
            landmarks = {
                "right_eye":   (int(kp[0, 0]), int(kp[0, 1])),
                "left_eye":    (int(kp[1, 0]), int(kp[1, 1])),
                "nose":        (int(kp[2, 0]), int(kp[2, 1])),
                "right_mouth": (int(kp[3, 0]), int(kp[3, 1])),
                "left_mouth":  (int(kp[4, 0]), int(kp[4, 1])),
            }
            faces.append({
                "x": x1, "y": y1, "w": w, "h": h,
                "score": float(scores[i]),
                "landmarks": landmarks,
            })
        return faces
