#BrollyCast☂️: A simple MLOps Pipeline for precipitation forecasting using ERA5 Data

An end-to-end Machine Learning pipeline for short-term precipitation forecasting. This project ingests raw meteorological netCDF data, trains a custom neural network using chronological time-series validation, and serves predictions via a containerised REST API.

Having lived in Ireland and the UK for almost 8 years, I know well the fear of not knowing when it is going to rain. With this simple prototype project, one can input 3 hours of meteorological data and obtain a probability of precipitation, perfect to know if you should get your umbrella (or brolly) out with you!

This project is also designed to showcase production-ready MLOps practices, rigorous meteorological evaluation metrics, and cloud-ready deployment strategies.

## Architecture and Tools
**Data Engineering:** xarray, pandas, ECMWF ERA5 (netCDF formats)

**Machine Learning:** PyTorch, scikit-learn

**Deployment & MLOps:** FastAPI, Docker (CPU-optimised inference)

## Pipeline Architecture
**Data Ingestion:** Extracts hourly readings (Temperature, Dewpoint, MSL Pressure, Cloud Cover) for a tight bounding box from raw ERA5 .nc files.

**Feature Engineering:** Computes a 3-hour rolling lag-window and scales features using a fitted StandardScaler to handle extreme atmospheric variance.

**Model Training:** A PyTorch Multi-Layer Perceptron (MLP) trained to predict the probability of precipitation (>0.01mm) in the following hour. Evaluated using a strict chronological train/test split to prevent temporal data leakage.

**API Serving:** The trained model weights and data scaler are loaded into a FastAPI application with strict Pydantic validation schemas.

**Containerisation:** Packaged into a lightweight Docker container explicitly configured for CPU-only PyTorch inference, minimizing image size and maximizing deployment flexibility.

## Model Performance (Meteorological Context)
Standard accuracy is a poor metric for weather forecasting due to high class imbalance. The model was evaluated on unseen future data (chronological split) with a focus on precision and F1-Score:

Overall Accuracy: 88%

Precipitation Precision: 87% (Low False-Alarm Rate)

Precipitation Recall: 79%

F1-Score: 0.83

*Note:* The model successfully identifies classic storm profiles (e.g., rapid pressure drops combined with 100% relative humidity). Future iterations would integrate spatial radar data to improve recall for isolated, localised showers.

## Quick Start: Run the API via Docker
You do not need to install PyTorch or download the massive ERA5 datasets to run this model. The trained weights and scaler are included, and the environment is fully containerised.

1. Clone the repository and navigate to the directory:

```
git clone https://github.com/anasdm/BrollyCast.git
cd BrollyCast
```

2. Build the Docker image:

```
docker build -t rain-api .
```
3. Run the container:

```
docker run -p 8080:8000 rain-api
```
4. Test the Model:

Open your browser and navigate to the interactive API documentation:
http://127.0.0.1:8080/docs

You can test the model's meteorological logic by sending a JSON payload via the POST /predict endpoint.
