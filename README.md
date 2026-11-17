# YOLO11 Object Detection

> Detect objects in images and videos with YOLO11 (Ultralytics), OpenCV and a Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)


## Features

- Object detection on uploaded images and videos.
- Bounding boxes with class labels and confidence.
- Streamlit UI for quick testing.
- Uses the small `yolo11n` model, so it runs on a laptop CPU.

## Tech Stack

| Layer | Technology |
|---|---|
| Detection | Ultralytics YOLO11 |
| Vision | OpenCV |
| UI | Streamlit |

## Quick Start

```bash
git clone https://github.com/dastonbek8747/yolo11-object-detection.git
cd yolo11-object-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

## Project Structure

```
main.py    # Streamlit app
yolo.py    # detection helpers
yolo11n.pt # pretrained weights
```

## License

MIT © 2026 Doston Sultonov.
