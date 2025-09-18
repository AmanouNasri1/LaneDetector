# Lane Detection with Deep Learning (TuSimple)

This project implements a **lane detection system** using the [TuSimple dataset](https://github.com/TuSimple/tusimple-benchmark), powered by a U-Net segmentation model.  
It supports **training, evaluation, test prediction, and video inference**.

---

## 📌 Features
- ✅ Dataset preprocessing: flatten TuSimple’s nested folder structure into paired `images/` and `masks/`.  
- ✅ Training pipeline with mixed precision for faster training on GPU.  
- ✅ Evaluation with loss metrics and overlay visualizations.  
- ✅ Test set predictions on TuSimple test data.  
- ✅ Video inference for real-world driving scenarios.  
- ✅ Reusable inference utilities (`inference.py`).  
- ✅ Baseline classical lane detection using Canny + Hough Transform for comparison.  

---

## 📂 Project Structure
LaneDetector/
│
├── src/
│ ├── deep_learning/
│ │ ├── dataset.py # TuSimple dataset loader
│ │ ├── model.py # U-Net implementation
│ │ ├── train.py # Training loop
│ │ ├── evaluate.py # Model evaluation
│ │ ├── predict.py # Predict on test set
│ │ ├── predict_video.py # Predict lanes on video
│ │ ├── inference.py # Single image inference utilities
│ │ ├── utils.py # Helpers (loss, transforms, etc.)
│ │ └── flatten_tusimple.py # Script to preprocess TuSimple dataset
│
├── videos/
│ ├── input/ # Raw input videos
│ └── output/ # Output lane overlay results
│
├── TUSimple/ # Dataset (not included in repo)
│ ├── train_set/
│ ├── test_set/
│ └── ...
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🚀 Getting Started

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Dataset preparation
Place the TuSimple dataset under TUSimple/ and run:

bash
Copy code
python -m src.deep_learning.flatten_tusimple
This will generate:

bash
Copy code
TUSimple/train_set/images/
TUSimple/train_set/masks/
3️⃣ Train the model
bash
Copy code
python -m src.deep_learning.train
4️⃣ Evaluate performance
bash
Copy code
python -m src.deep_learning.evaluate
5️⃣ Predict on the test set
bash
Copy code
python -m src.deep_learning.predict
6️⃣ Run video inference
Put a video in videos/input/ and run:

bash
Copy code
python -m src.deep_learning.predict_video
The result will be saved in videos/output/.

📊 Results
Training converges with Val Loss ≈ 0.05–0.06 after 20 epochs.

Lane overlays are reasonably accurate and robust to different lighting conditions.

Supports both real-time inference on images and videos.

🛠 Requirements
See requirements.txt.

✨ Future Work
Improve lane continuity using post-processing (e.g., spline fitting).

Add YOLO/PINet-based lane detection models for comparison.

Deploy a real-time demo using OpenCV or a web app.

🧑‍💻 Author

Project developed by Amanou Allah Nasri (adapted for personal use and experimentation).
📧 amanullah.nasri@outlook.com
🌐 [LinkedIn](https://www.linkedin.com/in/amanou-allah-nasri-6a5538260/)
📁 [GitHub Repo](https://github.com/AmanouNasri1/TrafficSign)