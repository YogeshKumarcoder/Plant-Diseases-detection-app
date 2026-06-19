# 🌿 CropGuard AI — Plant Disease Detection System

![CropGuard AI](frontend/public/screenshot.png)

## 🎯 Overview
An AI-powered plant disease detection system that analyzes leaf images and provides instant disease diagnosis with treatment recommendations.

## ✨ Features
- 🔍 **96.9% Accuracy** — EfficientNetB0 Transfer Learning
- 🌱 **38 Disease Classes** across 13 crop types
- 💊 **Treatment Recommendations** for every detected disease
- ⚡ **Real-time Analysis** — results in seconds
- 🖥️ **Production-ready** React + FastAPI stack

## 🏗️ Architecture

Leaf Image → FastAPI Backend → EfficientNetB0 Model → Disease + Treatment

↑                                                        ↓

React Frontend ←————————————— JSON Response ←————————————————

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | React.js |
| Backend | FastAPI + Uvicorn |
| ML Model | EfficientNetB0 (Transfer Learning) |
| Dataset | PlantVillage (87,000+ images) |
| Training | Kaggle GPU (T4) |

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| Validation Accuracy | **96.9%** |
| Training Images | 70,295 |
| Validation Images | 17,572 |
| Disease Classes | 38 |
| Epochs | 5 |

## 🌿 Supported Crops
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## 🚀 How to Run

### Backend
```bash
cd backend
pip install fastapi uvicorn tensorflow pillow python-multipart
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## 📁 Project Structure

cropguard-ai/

├── backend/

│   ├── main.py                    # FastAPI server

│   ├── plant_disease_model.keras  # Trained model

│   ├── class_names.json           # 38 class labels

│   └── treatment_map.json         # Treatment recommendations

├── frontend/

│   └── src/

│       └── App.js                 # React UI

└── README.md

## 🧠 ML Pipeline
1. **Data Loading** — ImageDataGenerator with batch processing
2. **Preprocessing** — EfficientNet-specific normalization
3. **Transfer Learning** — EfficientNetB0 pretrained on ImageNet (frozen)
4. **Custom Head** — GlobalAveragePooling → Dropout(0.3) → Dense(38)
5. **Training** — Adam optimizer, categorical crossentropy loss

## 👨‍💻 Author
**Yogesh Kumar** — Mechanical Engineering Student @ AMU
- Self-taught ML Engineer targeting FAANG
- GitHub: [@YogeshKumarcoder](https://github.com/YogeshKumarcoder)