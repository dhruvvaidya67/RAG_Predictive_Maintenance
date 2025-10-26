# RAG_Predictive_Maintenance
AI-Powered Predictive Maintenance System using LSTM Neural Networks and RAG for turbofan engine RUL prediction with interactive Streamlit dashboard


# 🔧 RAG-Based Predictive Maintenance System

An AI-powered predictive maintenance platform that combines **LSTM Neural Networks** with **Retrieval-Augmented Generation (RAG)** to predict Remaining Useful Life (RUL) of turbofan engines and provide intelligent maintenance guidance.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- **LSTM-Based RUL Prediction**: Deep learning model predicting equipment failure with 89.5% confidence
- **Interactive Dashboard**: Real-time predictions via Streamlit web interface
- **RAG Q&A System**: Intelligent maintenance knowledge base using vector embeddings
- **Multi-Sensor Analysis**: Processes 21 sensor readings including temperature, pressure, and vibration
- **Custom Maintenance Manual**: 12-page technical documentation for turbofan engines
- **Automated Pipeline**: End-to-end data processing from raw sensor data to predictions

## 🏗️ Architecture

Data Pipeline → Preprocessing → LSTM Model → RUL Prediction
↓
RAG System ← Knowledge Base → Q&A Interface


## 📊 Dataset

- **Source**: NASA C-MAPSS Turbofan Engine Degradation Dataset
- **Records**: 20,631 operational cycles
- **Sensors**: 21 operational settings and measurements
- **Use Case**: Aircraft engine health monitoring

## 🛠️ Tech Stack

- **ML Framework**: TensorFlow 2.20, Keras 3
- **NLP**: Sentence Transformers, LangChain
- **Vector Store**: FAISS
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Plotly

## 📁 Project Structure

RAG_Predictive_Maintenance/
├── data/
│ └── raw/ # NASA dataset files
├── src/
│ ├── data_preprocessing/ # Data loading & preprocessing
│ ├── model_training/ # LSTM model training
│ └── rag_system/ # RAG implementation
├── models/
│ ├── lstm/ # Trained LSTM models
│ └── rag/ # Vector store index
├── maintenance_manuals/
│ └── pdfs/ # Technical manuals
├── app/
│ └── streamlit_app.py # Web interface
└── README.md


## 🚀 Quick Start

### Prerequisites

Python 3.11
TensorFlow 2.20
Streamlit 1.39


### Installation

Train LSTM model
python -m src.model_training.train_pipeline

Build RAG vector store
python -m src.rag_system.build_vector_store
