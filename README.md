# Sentiment Analysis on Arabic Text with Comprehensive MLOps Integration

This project is designed to perform **sentiment analysis on Arabic text** using a pre-trained BERT-based model. It includes a **FastAPI backend** for serving predictions, a **Docker setup** for containerization, and a **training pipeline** to fine-tune the model on your dataset.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Training the Model](#training-the-model)
6. [Running the Application](#running-the-application)
7. [API Endpoints](#api-endpoints)
8. [Docker Integration](#docker-integration)
10. [License](#license)

---

## Project Overview

This project provides a robust pipeline for sentiment analysis on Arabic text. It leverages a pre-trained BERT-based model (`asafaya/bert-base-arabic`) to classify text into one of four sentiment categories: **Positive**, **Negative**, **Neutral**, or **Objective**. The project includes a **training script** to fine-tune the model on your dataset and a **FastAPI backend** for serving predictions.

---

## Features
- **Sentiment Analysis:** Classify Arabic text into four sentiment categories.
- **Training Pipeline:** Fine-tune the pre-trained BERT model on your dataset.
- **FastAPI Backend:** A RESTful API for serving predictions.
- **Docker Support:** Containerized application for easy deployment.
- **Logging and Monitoring:** Logs predictions and generates reports for monitoring.

---

## Technologies Used
- **Python**
- **FastAPI**: For building the REST API.
- **Hugging Face Transformers**: For using the pre-trained BERT model.
- **PyTorch**: For model training and inference.
- **Docker**: For containerizing the application.
- **Pydantic**: For data validation in the API.

---

## Setup and Installation

### Prerequisites
- Python 3.9 or higher
- Docker (optional, for containerization)
- Git (for version control)

### Steps

1. **Clone the Repository**
    ```bash
    git clone https://github.com/firesoul1422/Sentiment-Analysis-on-Arabic-Text-with-Comprehensive-MLOps-Integration.git
    cd Sentiment-Analysis-on-Arabic-Text-with-Comprehensive-MLOps-Integration
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```


---



## Training the Model

Before running the application, you should train (or fine-tune) the model using your dataset to optimize its performance.

### Prepare the Dataset

1.  Place your dataset in the `data/` directory (or the directory specified by `DATA_PATH` in your `.env` file).
2.  Ensure the dataset is in a suitable format (e.g., CSV or a text file with clearly labeled sentiment data). The expected format should have at least two columns: one for the text and one for the corresponding sentiment label.

### Run the Training Script

    ```bash
    python training.py <ddata_location>
    ```

*   The trained model will be saved in the `model/` directory.

---

## Running the Application

You can run the application either using Python directly or with Docker.

### Using Python

1.  **Start the FastAPI Server**

    ```bash
    uvicorn server:app --reload
    ```

    The `--reload` flag enables automatic server reloading upon code changes, which is useful during development.



### Using Docker

1.  **Build the Docker Image**

    ```bash
    docker build -t sentiment-analysis-api .
    ```

2.  **Run the Docker Container**

    ```bash
    docker run -p 8000:8000 sentiment-analysis-api
    ```

---

## API Endpoints

### 1. Home Page

*   **Endpoint:** `GET /`
*   **Description:** Returns the home page, which includes a simple user interface for testing the API. You can enter Arabic text and get its sentiment prediction.

### 2. Predict Sentiment

*   **Endpoint:** `POST /predict`
*   **Description:** Predicts the sentiment of the input Arabic text.
*   **Request Body:**

    ```json
    {
      "tweet": "النص العربي هنا"
    }
    ```

*   **Response:**

    ```json
    {
      "sentiment": "POS"
    }
    ```
    Where the `sentiment` field will contain one of the following:

    *   `POS` (Positive)
    *   `NEG` (Negative)
    *   `NEU` (Neutral)
    *   `OBJ` (Objective)

---

## Docker Integration

This project is fully containerized using Docker, simplifying deployment and ensuring consistent behavior across different environments. You can build and run the Docker image locally for development or testing, or deploy it to a container orchestration platform like Kubernetes for production.

### Build and Run

```bash
docker build -t sentiment-analysis-api .
docker run -p 8000:8000 sentiment-analysis-api
```



## License

*   This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## Acknowledgments

*   **Hugging Face** for providing the pre-trained BERT model (`asafaya/bert-base-arabic`) and the Transformers library.
*   **FastAPI** for the efficient and user-friendly API framework.

---

## Contact

For any questions, suggestions, or feedback, please feel free to contact:

*   **Name:** \ Mohammad Ataiq Alzahrani
*   **Email:** moai.ksa1@gmail.com
*   **GitHub:** firesoul1422

