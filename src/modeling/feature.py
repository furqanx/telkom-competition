import os
import torch
from torchvision import transforms

import numpy as np
import pandas as pd
from PIL import Image

class VideoPreprocessing:
    def __init__(self, resize_dim=(224, 224)):
        self.resize_dim = resize_dim
        
        # Definisi Transformasi Dasar (Resize & Normalisasi)
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_dim),
            transforms.ToTensor(), # Mengubah ke (C, H, W) dan range [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, frame_list):
        """
        Input: List of PIL Images (5 frames)
        Output: Tensor (3, 5, 224, 224) -> (Channel, Time, Height, Width)
        """
        processed_frames = []
        for img in frame_list:
            # Terapkan transformasi per frame
            tensor_img = self.transform(img)
            processed_frames.append(tensor_img)
        
        # Stack frames: (5, 3, 224, 224)
        video_tensor = torch.stack(processed_frames)
        
        # Ubah dimensi agar sesuai model 3D-CNN: (Channel, Time, Height, Width)
        # Dari (Time, Channel, H, W) -> (Channel, Time, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor

class FeatureEngineering:
    """
    Placeholder untuk teknik lanjutan.
    Misal: Anda ingin menambahkan Optical Flow atau filter khusus.
    """
    def __init__(self):
        pass

    def apply(self, video_tensor):

        # [PLACEHOLDER] Tulis logika feature engineering tambahan di sini
        # Contoh: Mengurangi noise, menaikkan kontras, dll.
        # Saat ini kita biarkan pass (return as is)
        
        ...
        
        return video_tensor