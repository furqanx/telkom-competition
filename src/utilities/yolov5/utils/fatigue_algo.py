import numpy as np
from collections import deque

class FatigueCalculator:
    """
    Implementasi Logika Bagian 3.2:
    - Hitung EAR (Eye Aspect Ratio)
    - Hitung MAR (Mouth Aspect Ratio)
    - Hitung PERCLOS & Continuous Frames
    """
    def __init__(self, unit_cycle=150):
        # Thresholds (Sesuai Paper Section 3.2.1)
        self.EAR_THRESH = 0.02  # Jika < 0.02 dianggap tutup mata
        self.MAR_THRESH = 0.65  # Jika > 0.65 dianggap menguap
        
        # Thresholds Keputusan (Sesuai Eq 14)
        self.PERCLOS_THRESH = 0.15 # 15% dari waktu
        self.FRAME_EYE_CLOSED_THRESH = 20
        self.FRAME_YAWN_THRESH = 30
        
        # History Buffer untuk PERCLOS (Sliding Window 150 frame)
        self.unit_cycle = unit_cycle
        self.eye_history = deque(maxlen=unit_cycle)   # 1 = Closed, 0 = Open
        self.mouth_history = deque(maxlen=unit_cycle) # 1 = Yawn, 0 = Normal
        
        # Counter untuk Continuous Frames (Eq 10 & 11)
        self.cnt_frame_eye = 0
        self.cnt_frame_yawn = 0

    def calculate_distance(self, p1, p2):
        """Euclidean distance antara dua titik 3D/2D"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def calculate_ear(self, landmarks):
        """
        Menghitung Eye Aspect Ratio (EAR) berdasarkan Eq (6), (7), (8).
        Landmarks harus list of tuples (x, y).
        Indeks sesuai Figure 10 Paper.
        """
        # Titik Mata Kanan (Right Eye)
        # Vertical 1: 384 - 381
        # Vertical 2: 386 - 374
        # Vertical 3: 388 - 390
        # Horizontal: 362 - 263
        v1_r = self.calculate_distance(landmarks[384], landmarks[381])
        v2_r = self.calculate_distance(landmarks[386], landmarks[374])
        v3_r = self.calculate_distance(landmarks[388], landmarks[390])
        h_r  = self.calculate_distance(landmarks[362], landmarks[263])
        ear_right = (v1_r + v2_r + v3_r) / (3.0 * h_r + 1e-6)

        # Titik Mata Kiri (Left Eye)
        # Vertical 1: 161 - 163
        # Vertical 2: 159 - 145
        # Vertical 3: 157 - 154
        # Horizontal: 33 - 133
        v1_l = self.calculate_distance(landmarks[161], landmarks[163])
        v2_l = self.calculate_distance(landmarks[159], landmarks[145])
        v3_l = self.calculate_distance(landmarks[157], landmarks[154])
        h_l  = self.calculate_distance(landmarks[33], landmarks[133])
        ear_left = (v1_l + v2_l + v3_l) / (3.0 * h_l + 1e-6)

        # Average (Eq 8)
        return (ear_left + ear_right) / 2.0

    def calculate_mar(self, landmarks):
        """
        Menghitung Mouth Aspect Ratio (MAR) berdasarkan Eq (9).
        """
        # Vertical 1: 39 - 181
        # Vertical 2: 0 - 17
        # Vertical 3: 269 - 405
        # Horizontal: 61 - 291
        v1 = self.calculate_distance(landmarks[39], landmarks[181])
        v2 = self.calculate_distance(landmarks[0], landmarks[17])
        v3 = self.calculate_distance(landmarks[269], landmarks[405])
        h  = self.calculate_distance(landmarks[61], landmarks[291])
        
        mar = (v1 + v2 + v3) / (3.0 * h + 1e-6)
        return mar

    def update_status(self, ear, mar, yolo_eyes_closed=False, yolo_mouth_open=False):
        """
        Update status per frame dan hitung logika Fatigue.
        Menggabungkan hasil Geometri (EAR/MAR) dan YOLO (Eq 15).
        """
        
        # --- 1. Fusion Logic (Eq 15) ---
        # Mata dianggap tutup jika: YOLO bilang tutup ATAU EAR < Threshold
        is_eye_closed = yolo_eyes_closed or (ear < self.EAR_THRESH)
        
        # Mulut dianggap menguap jika: YOLO bilang buka DAN MAR > Threshold
        # (Catatan: Paper bilang DAN untuk mulut, ATAU untuk mata)
        is_yawning = yolo_mouth_open and (mar > self.MAR_THRESH)
        # Fallback: Jika YOLO tidak deteksi tapi MAR sangat besar, anggap Yawn
        if mar > 0.75: is_yawning = True 

        # --- 2. Update Counters (Eq 10 & 11) ---
        if is_eye_closed:
            self.cnt_frame_eye += 1
            self.eye_history.append(1)
        else:
            self.cnt_frame_eye = 0
            self.eye_history.append(0)

        if is_yawning:
            self.cnt_frame_yawn += 1
            self.mouth_history.append(1)
        else:
            self.cnt_frame_yawn = 0
            self.mouth_history.append(0)

        # --- 3. Hitung PERCLOS (Eq 12 & 13) ---
        # PERCLOS = Jumlah frame tutup / Total frame window (150)
        perclos_eyes = sum(self.eye_history) / self.unit_cycle
        perclos_mouth = sum(self.mouth_history) / self.unit_cycle

        # --- 4. Keputusan Akhir (Eq 14) ---
        is_fatigue = False
        reason = "Normal"

        if perclos_eyes >= self.PERCLOS_THRESH:
            is_fatigue = True
            reason = "PERCLOS Eyes High"
        elif perclos_mouth >= self.PERCLOS_THRESH:
            is_fatigue = True
            reason = "PERCLOS Mouth High"
        elif self.cnt_frame_eye >= self.FRAME_EYE_CLOSED_THRESH:
            is_fatigue = True
            reason = "Microsleep Detected"
        elif self.cnt_frame_yawn >= self.FRAME_YAWN_THRESH:
            is_fatigue = True
            reason = "Continuous Yawning"

        return is_fatigue, reason, (ear, mar, perclos_eyes, perclos_mouth)