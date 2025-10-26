
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
│   ├── raw/                    # NASA dataset files
│   ├── processed/              # Preprocessed data outputs
│   └── manuals/                # Technical PDF manuals
├── src/
│   ├── data_preprocessing/     # Data loading & preprocessing scripts
│   ├── model_training/         # LSTM model training code
│   └── rag_system/             # RAG implementation
├── models/
│   └── lstm/                   # Trained LSTM models
├── app/
│   └── streamlit_app.py        # Web interface (or keep in root)
└── README.md                   # Project documentation



## 🚀 Quick Start

### Prerequisites

Python 3.11
TensorFlow 2.20
Streamlit 1.39


### Installation

Clone repository
git clone https://github.com/yourusername/RAG_Predictive_Maintenance.git
cd RAG_Predictive_Maintenance

Install dependencies
pip install -r requirements.txt

Download NASA dataset
Place train_FD001.txt in data/raw/


### Training the Model

Train LSTM model
python -m src.model_training.train_pipeline

Build RAG vector store
python -m src.rag_system.build_vector_store


### Running the Application

streamlit run app/streamlit_app.py


Access the dashboard at `http://localhost:8501`

## 💻 Usage

### 1. RUL Prediction

- Navigate to **"RUL Prediction"** tab
- Click **"Load Model"**
- Click **"Predict RUL"** to get remaining useful life estimation
- View confidence score and health status

### 2. Maintenance Q&A

- Navigate to **"Maintenance Q&A System"** tab
- Click **"Load Knowledge Base"**
- Ask questions about maintenance procedures
- Get AI-powered answers from technical manuals

### 3. System Status

- View model performance metrics
- Check system health indicators
- Monitor prediction accuracy

## 📈 Model Performance

- **Architecture**: 2-layer LSTM with Dropout regularization
- **Input**: 50 timesteps × 21 features
- **Training**: 50 epochs, Adam optimizer
- **Validation MAE**: ~15 cycles
- **Prediction Confidence**: 85-95%

## 🔬 Key Components

### LSTM Model

Layer 1: LSTM(50 units, return_sequences=True)

Dropout: 0.2

Layer 2: LSTM(50 units)

Dropout: 0.2

Output: Dense(1) - RUL prediction


### RAG System

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector Store: FAISS index

Document Loader: PyPDF for maintenance manuals

Chunk Size: 500 characters with 50 overlap



## 📝 Features in Detail

### Sensor Monitoring

- **Temperature Sensors**: Fan inlet, LPC, HPC, LPT
- **Pressure Sensors**: Static pressure, bypass ratio
- **Performance Metrics**: Total temperature, enthalpy
- **Operational Settings**: Altitude, Mach number, throttle

### Intelligent Q&A

- Natural language queries
- Context-aware responses
- Source citation from manuals
- Real-time answer generation

## 🎨 Screenshots

![Dashboard](screenshots/dashboard.png)
![Predictions](screenshots/predictions.png)
![RAG Q&A](screenshots/rag_qa.png)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NASA for the C-MAPSS dataset
- Hugging Face for transformer models
- Streamlit for the amazing framework

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 🔮 Future Enhancements

- [ ] Multi-engine type support
- [ ] Real-time sensor data integration
- [ ] Anomaly detection alerts
- [ ] Mobile application
- [ ] API for external integrations
- [ ] Advanced visualization dashboard

---

**Built with ❤️ using TensorFlow and Streamlit**

📝 REQUIREMENTS.TXT
Create this file:


