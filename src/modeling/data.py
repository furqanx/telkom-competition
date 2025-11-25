import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from src.modeling.feature import FeatureEngineering

class DriverDrowsinessDataset(Dataset):
    def __init__(self, root_dir, annotation_file, sequence_length=5, transform=None):
        """
        Args:
            root_dir (str): Path ke folder gambar.
            annotation_file (str): Path ke file CSV.
            sequence_length (int): Jumlah frame per klip (Paper = 5).
            transform (callable): Fungsi preprocessing.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.feature_engineering = FeatureEngineering()
        
        # Load Annotations
        self.df = pd.read_csv(annotation_file)
        
        # --- MENYIAPKAN DAFTAR KLIP (CLIPS) ---
        # Kita harus mengelompokkan data per video, lalu memotongnya jadi klip 5 frame.
        self.samples = self._make_clips()

    def __len__(self):
        return len(self.samples)

    def _make_clips(self):
        """
        Membuat daftar indeks awal untuk setiap klip.
        Contoh: Video A punya 10 frame. Sequence=5.
        Clips: [Frame 0-4], [Frame 5-9] (Non-overlapping) 
        atau [Frame 0-4], [Frame 1-5]... (Sliding window).
        
        Di sini saya gunakan Non-overlapping agar data tidak redundan (sesuai 'sets of small video clip').
        """
        clips = []
        
        # Grouping berdasarkan Video ID
        video_groups = self.df.groupby('video_id')
        
        for vid_id, group in video_groups:
            # Urutkan berdasarkan frame_id
            group = group.sort_values('frame_id').reset_index(drop=True)
            total_frames = len(group)
            
            # Buat klip per 5 frame
            for i in range(0, total_frames - self.sequence_length + 1, self.sequence_length):
                # Simpan metadata klip: (Video_ID, Start_Index_in_Group)
                clips.append((vid_id, i, group.iloc[i : i+self.sequence_length]))
                
        return clips

    def _aggregate_labels(self, clip_df):
        """
        "Assign annotation value which is observed more than 3 in each clip"
        (Mayoritas > 50% dari 5 frame)
        """
        # Hitung mode (nilai yang paling sering muncul) untuk setiap kolom label
        modes = clip_df.mode().iloc[0]
        
        # Ambil label (sesuaikan nama kolom dengan CSV anda)
        drowsiness = int(modes['drowsiness_label'])
        
        # Scene Labels (One-Hot Logic akan dihandle di sini atau di model)
        # Sederhananya kita ambil integer kategorinya dulu
        glasses = int(modes['glasses_label'])
        head = int(modes['head_label'])
        mouth = int(modes['mouth_label'])
        eye = int(modes['eye_label'])
        
        # Konversi ke Format Vector Lc (13 Dimensi)
        # Urutan: Glasses(5), Head(3), Mouth(3), Eye(2)
        lc_vector = torch.zeros(13)
        
        # One-hot encoding manual
        # Asumsi label di CSV berupa index (0,1,2..)
        lc_vector[0 + glasses] = 1.0  # Glasses offset 0
        lc_vector[5 + head] = 1.0     # Head offset 5
        lc_vector[8 + mouth] = 1.0    # Mouth offset 8
        lc_vector[11 + eye] = 1.0     # Eye offset 11
        
        return torch.tensor(drowsiness).long(), lc_vector

    def __getitem__(self, idx):
        video_id, start_idx, clip_df = self.samples[idx]
        
        images = []
        # Loop untuk memuat 5 gambar
        for _, row in clip_df.iterrows():
            frame_name = row['frame_id'] # misal: 'frame_001.jpg'
            img_path = os.path.join(self.root_dir, video_id, frame_name)
            
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except FileNotFoundError:
                # Fallback jika gambar rusak/hilang (isi dengan frame hitam atau duplikat)
                print(f"Warning: {img_path} not found.")
                img = Image.new('RGB', (224, 224))
                images.append(img)
        
        # 1. preprocessing (resize, normalize, stack -> tensor)
        if self.transform:
            video_tensor = self.transform(images)
        
        # 2. feature engineering (optional placeholder)
        video_tensor = self.feature_engineering.apply(video_tensor)
        
        # 3. aggregasi label (IOU logic)
        drowsiness_label, scene_label_vector = self._aggregate_labels(clip_df)
        
        return video_tensor, scene_label_vector, drowsiness_label