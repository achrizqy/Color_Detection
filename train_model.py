# FINAL TRAINER - TRAINING DARI FOLDER DATASET

import cv2
import numpy as np
import os
import pickle
from collections import Counter

# === AMBIL LABEL WARNA OTOMATIS DARI NAMA FOLDER ===
def get_label_from_hue(hsv_crop):
    hue_channel = hsv_crop[:, :, 0].flatten()
    hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))
    return dominant_hue, hist

# === MEMBANGUN PETA LABEL -> HUE DARI DATASET ===
def build_hue_map_from_dataset(dataset_path):
    label_hue_map = {}
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            hue_list = []
            print(f"🔍 Memproses label: {label}")
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Gagal membaca gambar: {img_path}")
                    continue
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                dominant_hue, _ = get_label_from_hue(hsv)
                hue_list.append(dominant_hue)
            if hue_list:
                most_common_hue = Counter(hue_list).most_common(1)[0][0]
                label_hue_map[label] = most_common_hue
                print(f"✅ Dominan HUE {most_common_hue} untuk label '{label}'")
            else:
                print(f"❌ Tidak ada gambar valid di label: {label}")
    return label_hue_map

# === JALANKAN TRAINING ===
DATASET_FOLDER = "dataset"
hue_label_map = build_hue_map_from_dataset(DATASET_FOLDER)

# Simpan hasil ke file
with open('hue_label_map.pkl', 'wb') as f:
    pickle.dump(hue_label_map, f)

print("\n✅ TRAINING SELESAI. Hasil disimpan di 'hue_label_map.pkl'")
