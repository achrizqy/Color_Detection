# 🚗🎨 Deteksi Warna Mobil Menggunakan MobileNet SSD + LSTM/HSV

Proyek ini adalah implementasi AI untuk **mendeteksi objek mobil** pada video atau webcam, kemudian **mengklasifikasikan warna dominan** pada mobil tersebut.  
Sistem dapat menerima input dari webcam maupun file video (`.mp4`) dan memberikan output berupa **bounding box** serta **label warna dominan**.

---

## ✨ Fitur
✅ Deteksi objek mobil menggunakan **MobileNet SSD** (OpenCV DNN)  
✅ Klasifikasi warna dominan menggunakan **analisis HSV** atau model **LSTM/CNN** yang sudah dilatih  
✅ Input bisa dari **webcam** atau **video file (.mp4)**  
✅ Output berupa tampilan real-time atau file `result_detection.mp4`

---

## 📂 Struktur Folder yang Direkomendasikan
├── color_trainer.py✅
├── color_detector_runtime.py✅
├── MobileNetSSD_deploy.prototxt✅
├── MobileNetSSD_deploy.caffemodel✅
├── hue_label_map.pkl ✅
├── result_detection.mp4 # hasil deteksi warna
├── dataset/ (silahakn download pada Gdrive)

---

## 🔗 Dataset dan Sample Video
📁 **Dataset warna mobil**: [👉 Google Drive Link Dataset](https://drive.google.com/drive/folders/1SS7-S_2WO-1jxuVXOhKPxcj3BgXmn-TD?usp=sharing)  
🎞️ **Sample Video input**: [👉 Google Drive Link Dataset](https://drive.google.com/drive/folders/1SS7-S_2WO-1jxuVXOhKPxcj3BgXmn-TD?usp=sharing)  

> Unduh dataset & sample video dari link di atas, lalu letakkan di folder proyekmu:
> ```
> dataset    # untuk dataset
> sample_video.mp4 # untuk input deteksi
> ```

---

## 🚀 Cara Menjalankan

### 1️⃣ Persiapan Environment
Buat virtual environment dan instal dependensi:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install opencv-python numpy
# Jika pakai CNN/LSTM:
pip install tensorflow keras

2️⃣ Training Warna
Pastikan folder dataset/ sudah diisi. Jalankan:
python color_trainer.py
✅ Hasil: file hue_label_map.pkl

3️⃣ Jalankan Detector
python color_detector_runtime.py

Pilih input:

1 untuk webcam
2 untuk file video (sample_video.mp4)

✅ Output:

Bounding box mobil + label warna pada jendela

Jika pakai input video, hasilnya juga tersimpan di result_detection.mp4

