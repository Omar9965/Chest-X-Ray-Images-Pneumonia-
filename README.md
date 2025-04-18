
# 🩺 Chest X-Ray Pneumonia Classifier API

A FastAPI-powered web service that classifies chest X-ray images as **Pneumonia** or **Normal** using a PyTorch DenseNet121 model.

---

## 🚀 Features

- Deep learning model (DenseNet121) trained for pneumonia detection.
- REST API built with FastAPI.
- Image upload endpoint for real-time inference.
- Swagger UI for interactive API docs.

---

## 🧠 Model

- Architecture: `DenseNet121`
- Custom classifier head:  
  `Dropout(0.2) + Linear -> Sigmoid`
- Output: Binary classification (`Pneumonia` vs. `Normal`)

---

## 📁 Project Structure

```
.
├── app.py              # FastAPI server
├── infer.py            # Model loading and inference logic
├── model/
│   └── best_model.pth  # Trained PyTorch model checkpoint
├── template/
│   └── index.html
├── images/
    └── Normal
    └── Pneumonia
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Omar9965/Chest-X-Ray-Images-Pneumonia-
cd chest-xray-api
```


### 2. Start the API Server

```bash
uvicorn app:app --reload
```

> The API will be available at `http://127.0.0.1:8000`

---

## 📸 Usage

### 🔬 Predict from Swagger UI

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

1. Click on **POST `/predict/`**
2. Upload a chest X-ray image
3. Get prediction and probability

### 📡 Predict via cURL

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -F 'file=@path_to_image.jpeg' \
  -H 'accept: application/json'
```

---

## 🛠 API Endpoints

### `GET /`

Returns a welcome message.

### `POST /predict/`

Upload a chest X-ray image (JPEG/PNG) and receive a prediction.

**Response:**
```json
{
  "prediction": "Pneumonia",
  "probability": 0.8374
}
```

---




## 🙌 Acknowledgments

- Dataset: [Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- PyTorch & FastAPI communities
