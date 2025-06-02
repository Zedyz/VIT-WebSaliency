import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import USE_FACE_MASK, USE_TEXT_MASK, USE_BANNER_MASK


class Dataloader(Dataset):
    """
    returns each sample as:
      - "image_6ch":  [6,720,1280] => (RGB + face + text + banner)
      - "face_full":  [1,720,1280]
      - "text_full":  [1,720,1280]
      - "banner_full":[1,720,1280]
      - "fdm_full":   [1,720,1280]
      - "eye_full":   [1,720,1280]
    """

    def __init__(self, root, file_list):
        super().__init__()
        self.root = root
        self.files = file_list

        self.img_dir = os.path.join(root, "dataset/images")
        self.face_dir = os.path.join(root, "dataset/face_mask")
        self.text_dir = os.path.join(root, "dataset/text_mask")
        self.banner_dir = os.path.join(root, "dataset/banner_mask")

        self.fdm_dir = os.path.join(root, "dataset/fdm_full")
        self.eye_dir = os.path.join(root, "dataset/eye_fixations")

    def __len__(self):
        return len(self.files)

    def load_gray_0to1(self, path):
        if not os.path.exists(path):
            return np.zeros((720, 1280), np.float32)
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            return np.zeros((720, 1280), np.float32)
        if raw.shape != (720, 1280):
            raw = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
        return raw.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        fname = self.files[idx]

        img_path = os.path.join(self.img_dir, fname)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((720, 1280, 3), np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        def load_mask_if_enabled(dir_, fname_):
            if not dir_ or not os.path.exists(dir_):
                return np.zeros((720, 1280), np.float32)
            fullp = os.path.join(dir_, fname_)
            return self.load_gray_0to1(fullp)

        face_np = load_mask_if_enabled(self.face_dir, fname) if USE_FACE_MASK else np.zeros((720, 1280), np.float32)
        text_np = load_mask_if_enabled(self.text_dir, fname) if USE_TEXT_MASK else np.zeros((720, 1280), np.float32)
        bann_np = load_mask_if_enabled(self.banner_dir, fname) if USE_BANNER_MASK else np.zeros((720, 1280), np.float32)

        rgb_ch = torch.from_numpy(rgb.transpose(2, 0, 1))
        face_ch = torch.from_numpy(face_np[None, ...])
        text_ch = torch.from_numpy(text_np[None, ...])
        bann_ch = torch.from_numpy(bann_np[None, ...])

        img_6ch = torch.cat([rgb_ch, face_ch, text_ch, bann_ch], dim=0)

        fdm_path = os.path.join(self.fdm_dir, fname)
        eye_path = os.path.join(self.eye_dir, fname)
        fdm_720 = self.load_gray_0to1(fdm_path)
        eye_720 = self.load_gray_0to1(eye_path)
        fdm_t = torch.from_numpy(fdm_720[None, ...])
        eye_t = torch.from_numpy(eye_720[None, ...])

        return {
            "stem": fname,
            "image_6ch": img_6ch,
            "face_full": face_ch,
            "text_full": text_ch,
            "banner_full": bann_ch,
            "fdm_full": fdm_t,
            "eye_full": eye_t
        }
