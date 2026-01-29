# Waste Segregation Image Classification Web App

This project is a simple end-to-end computer vision web application that classifies waste images into six categories using a deep learning model.

**Classes supported:**

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

Users can upload single or multiple images, and the app predicts the waste type along with confidence scores.

## Tech Stack

* **Backend:** FastAPI
* **Model:** PyTorch (TorchScript for deployment)
* **Frontend:** HTML, CSS, JavaScript
* **Image Processing:** PIL, torchvision

## Features

* Upload single or batch images for classification
* Fast and lightweight REST API using FastAPI
* Confidence score shown for every prediction
* Clean, simple web interface
* Supports CPU and GPU inference

## Setup Instructions

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
uvicorn main:app --reload
```

4. Open in browser:

```
http://127.0.0.1:8000/
```

## API Endpoint

**POST** `/predict`
Upload one or more images using form-data:

| Field | Type    | Description             |
| ----- | ------- | ----------------------- |
| files | File(s) | Image files to classify |

Returns:

```json
[
  {
    "class_id": 3,
    "class_name": "paper",
    "confidence": 0.87
  }
]
```

## What This Project Shows

* Ability to deploy a trained deep learning model
* Real-world usage of PyTorch in production
* API development with FastAPI
* Frontend-backend integration
* Handling batch data and inference pipelines

