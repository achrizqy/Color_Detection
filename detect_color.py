# FINAL DETECTOR UNTUK DETEKSI WARNA DOMINAN MOBIL BERDASARKAN HASIL TRAINER OTOMATIS

import cv2
import numpy as np
import pickle

# === LOAD MAPPING HUE -> LABEL YANG SUDAH DILATIH ===
with open('hue_label_map.pkl', 'rb') as f:
    hue_label_map = pickle.load(f)

# === LOAD MODEL DETEKSI MOBIL ===
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt',
    'MobileNetSSD_deploy.caffemodel'
)
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

# === FUNGSI PEMETAAN HUE KE LABEL (DARI FILE) ===
def map_hue_to_label(hue_dataset_map, hue_val):
    closest_label = None
    min_diff = 180
    for label, hue in hue_dataset_map.items():
        diff = abs(hue - hue_val)
        if diff < min_diff:
            min_diff = diff
            closest_label = label
    return closest_label

# === PILIH SUMBER INPUT ===
print("Pilih sumber input:")
print("1. Webcam (kamera index 1)")
print("2. Video file (mp4)")
pilihan = input("Masukkan 1 atau 2: ")

if pilihan == "1":
    cap = cv2.VideoCapture(1)
    save_output = False
elif pilihan == "2":
    path = input("Masukkan path video (contoh: input_video.mp4): ")
    cap = cv2.VideoCapture(path)
    save_output = True
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('result_detection.mp4', fourcc, fps, (width, height))
else:
    print("Input tidak valid.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if class_names[idx] == "car" and confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            car_img = frame[startY:endY, startX:endX]

            if car_img.shape[0] > 0 and car_img.shape[1] > 0:
                hsv = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)
                hue_vals = hsv[:, :, 0].flatten()
                hist = cv2.calcHist([hue_vals], [0], None, [180], [0, 180])
                dominant_hue = int(np.argmax(hist))
                label = map_hue_to_label(hue_label_map, dominant_hue)
                conf_pct = (hist[dominant_hue][0] / np.sum(hist)) * 100
                text_label = f"{label.upper()} ({conf_pct:.1f}%)"

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text_label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if save_output:
        out.write(frame)

    cv2.imshow("Deteksi Warna Dominan Mobil", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
