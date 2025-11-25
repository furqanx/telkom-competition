import cv2
import torch
import mediapipe as mp
import numpy as np
from pathlib import Path
import sys

# Tambahkan path agar bisa import dari src/utilities
sys.path.append('./src/utilities/yolov5') 

# Import Modul Buatan
from utils.fatigue_algo import FatigueCalculator

# Import YOLO modules (untuk load model nanti)
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox

def run():
    # --- KONFIGURASI ---
    weights = 'runs/train/exp8/weights/best.pt' # Path ke model hasil training Anda
    device_str = '' # '0' untuk GPU, 'cpu' untuk CPU
    
    # --- 1. Init MediaPipe Face Mesh (Attention Mesh) ---
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True, # Penting untuk iris/detail mata
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # --- 2. Init Logic Calculator ---
    fatigue_calc = FatigueCalculator()
    
    # --- 3. Init YOLOv5 Model ---
    device = select_device(device_str)
    try:
        model = DetectMultiBackend(weights, device=device)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(640, s=stride)
        print("Model YOLOv5 berhasil di-load!")
        has_yolo = True
    except Exception as e:
        print(f"Warning: Gagal load model YOLO ({e}). Berjalan mode MediaPipe Only.")
        has_yolo = False
        names = []

    # --- 4. Buka Kamera ---
    cap = cv2.VideoCapture(0) # 0 untuk webcam bawaan
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Copy frame untuk display
        display_frame = frame.copy()
        h, w, _ = frame.shape
        
        # ---------------------------------------------------------
        # STEP A: DETEKSI YOLOv5 (Untuk Klasifikasi Mata/Mulut)
        # ---------------------------------------------------------
        yolo_eyes_closed = False
        yolo_mouth_open = False
        
        if has_yolo:
            # Preprocess image
            img = letterbox(frame, imgsz, stride=stride)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            if len(img.shape) == 3: img = img[None]
            
            # Inference
            pred = model(img)
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, max_det=1000)
            
            # Process detections
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes to original image
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        label = names[int(cls)]
                        # Logic mapping label YOLO ke status (Sesuaikan nama class di data.yaml Anda)
                        if 'c_eyes' in label: yolo_eyes_closed = True
                        if 'o_mouth' in label: yolo_mouth_open = True
                        
                        # Gambar Box YOLO
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(display_frame, c1, c2, (255, 0, 0), 2)
                        cv2.putText(display_frame, label, (c1[0], c1[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # ---------------------------------------------------------
        # STEP B: MEDIAPIPE (Untuk 3D Keypoints & EAR/MAR)
        # ---------------------------------------------------------
        # Konversi ke RGB untuk MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        current_ear, current_mar = 0.0, 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Ambil koordinat 468 titik (Denormalize ke pixel)
                lm_points = []
                for lm in face_landmarks.landmark:
                    lm_points.append((int(lm.x * w), int(lm.y * h)))
                
                # Hitung EAR & MAR
                current_ear = fatigue_calc.calculate_ear(lm_points)
                current_mar = fatigue_calc.calculate_mar(lm_points)
                
                # Visualisasi Titik Mata & Mulut (Opsional)
                # Gambar titik indeks 33 (mata kiri) dsb... (Bisa ditambahkan loop)

        # ---------------------------------------------------------
        # STEP C: LOGIKA KEPUTUSAN (FUSION)
        # ---------------------------------------------------------
        is_fatigue, reason, stats = fatigue_calc.update_status(
            current_ear, current_mar, yolo_eyes_closed, yolo_mouth_open
        )
        ear, mar, perclos_e, perclos_m = stats

        # ---------------------------------------------------------
        # STEP D: TAMPILAN GUI
        # ---------------------------------------------------------
        # Warna status
        status_color = (0, 255, 0) if not is_fatigue else (0, 0, 255)
        status_text = "NORMAL" if not is_fatigue else f"FATIGUE: {reason}"
        
        # Info Panel
        cv2.rectangle(display_frame, (0, 0), (300, 160), (0, 0, 0), -1)
        cv2.putText(display_frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display_frame, f"EAR: {ear:.3f} (Thresh: 0.02)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"MAR: {mar:.3f} (Thresh: 0.65)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"PERCLOS Eye: {perclos_e:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display_frame, f"PERCLOS Mouth: {perclos_m:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if yolo_eyes_closed:
            cv2.putText(display_frame, "YOLO: Eyes Closed", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if yolo_mouth_open:
            cv2.putText(display_frame, "YOLO: Mouth Open", (320, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Fatigue Detection System', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()