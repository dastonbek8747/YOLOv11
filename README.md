# 🎯 YOLOv11 — Obyektlarni Aniqlash

**Ultralytics YOLO11** modeli yordamida rasm va videolarda obyektlarni aniqlovchi loyiha. Ikki usulda ishlaydi: lokal kamera (OpenCV) va Streamlit veb-interfeys.

---

## ✨ Imkoniyatlar

- 📷 **Real vaqt:** veb-kamera orqali jonli aniqlash
- 🎬 **Video fayl:** tayyor videoni qayta ishlash
- 🌐 **Streamlit interfeys:** brauzer orqali boshqarish
- 🏷 **80+ toifa:** YOLO11 standart toifalari (odam, mashina, hayvon va boshqalar)

---

## 🛠 Texnologiyalar

| Texnologiya | Maqsad |
|---|---|
| **Python 3.10+** | Asosiy til |
| **ultralytics** | YOLO11 modeli |
| **OpenCV** | Kamera/video |
| **Streamlit** | Veb-interfeys |

---

## ⚙️ O'rnatish

```bash
git clone https://github.com/dastonbek8747/YOLOv11.git
cd YOLOv11
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Model fayli (`yolo11n.pt`) repoda mavjud. Agar yo'q bo'lsa, birinchi ishga tushirishda avtomatik yuklanadi.

---

## ▶️ Ishga tushirish

**Veb-interfeys (video fayl bilan):**

```bash
streamlit run yolo.py
```

> `yolo.py` faylda `video_path = "video.mp4"` qatorini o'z videongiz nomiga o'zgartiring.

**Lokal kamera (OpenCV):**

```bash
python main.py
```

Chiqish uchun `1` tugmasini bosing.

---

## 📁 Loyiha tuzilishi

```
YOLOv11/
├── main.py            # OpenCV kamera namunasi
├── yolo.py            # Streamlit + video tahlil
├── yolo11n.pt         # Model fayli
├── img.png
├── requirements.txt
└── tests/
```

---

## ⚠️ Eslatma

Katta binar fayllar (`yolo11n.pt` ~5.4 MB, `img.png` ~4.5 MB) repoda saqlanadi.

---

## 📄 Litsenziya

Litsenziya ko'rsatilmagan.
