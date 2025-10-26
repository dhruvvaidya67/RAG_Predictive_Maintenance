"""
Complete Training Pipeline for RAG-Based Predictive Maintenance

This script orchestrates the entire data processing and model training pipeline:
1. Loads raw C-MAPSS data
2. Preprocesses the data (calculates RUL, adds rolling features, normalizes)
3. Prepares sequences for LSTM training
4. Splits data into train/validation sets
5. Builds and trains the LSTM model
6. Saves the trained model

Usage:
    python train_pipeline.py

Author: RAG Predictive Maintenance Team
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import custom modules
try:
    from src.data_preprocessing.load_data import load_cmapss_data
    from src.data_preprocessing.preprocess import preprocess_data
    from src.model_training.lstm_model import (
        prepare_sequences,
        build_lstm_model,
        train_model,
        save_model,
        evaluate_model
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all source files are in the correct directories.")
    sys.exit(1)


def main():
    """
    Main training pipeline for LSTM-based predictive maintenance.
    
    This function orchestrates the complete workflow:
    1. Load raw C-MAPSS data
    2. Preprocess data (RUL calculation, rolling features, normalization)
    3. Prepare sequences for LSTM training
    4. Split data into training and validation sets
    5. Build LSTM model architecture
    6. Train the model
    7. Save the trained model
    8. Display results and metrics
    """
    
    print("\n" + "="*80)
    print("RAG-Based Predictive Maintenance - Training Pipeline")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Load NASA C-MAPSS dataset")
    print("  2. Preprocess the data")
    print("  3. Train an LSTM model to predict RUL")
    print("  4. Save the trained model")
    print("="*80 + "\n")
    
    try:
        # ============================================================================
        # STEP 1: LOAD RAW DATA
        # ============================================================================
        print("\n" + "="*80)
        print("STEP 1: Loading Raw Data")
        print("="*80)
        
        print("\nLoading C-MAPSS dataset from data/raw/train_FD001.txt...")
        raw_data = load_cmapss_data()
        
        print(f"✓ Raw data loaded successfully!")
        print(f"  Shape: {raw_data.shape}")
        print(f"  Engines: {raw_data['unit_number'].nunique()}")
        
        # ============================================================================
        # STEP 2: PREPROCESS DATA
        # ============================================================================
        print("\n" + "="*80)
        print("STEP 2: Preprocessing Data")
        print("="*80)
        
        print("\nPreprocessing pipeline includes:")
        print("  - Calculating Remaining Useful Life (RUL)")
        print("  - Adding rolling window features (mean, std)")
        print("  - Normalizing sensor values")
        print("  - Handling missing values")
        
        # Preprocess the data
        # Set window_size=5 for rolling features, save_file=True to save to disk
        processed_data = preprocess_data(raw_data, window_size=5, save_file=True)
        
        print(f"\n✓ Data preprocessing complete!")
        print(f"  Shape after preprocessing: {processed_data.shape}")
        print(f"  Total features: {processed_data.shape[1]}")
        
        # ============================================================================
        # STEP 3: PREPARE SEQUENCES FOR LSTM
        # ============================================================================
        print("\n" + "="*80)
        print("STEP 3: Preparing Sequences for LSTM Training")
        print("="*80)
        
        # Define sequence length (number of timesteps to look back)
        sequence_length = 50
        
        print(f"\nCreating sequences with lookback window: {sequence_length}")
        print("This converts the time-series data into sequences suitable for LSTM")
        
        # Prepare sequences
        # This converts the data into (samples, timesteps, features) format
        X, y = prepare_sequences(processed_data, sequence_length=sequence_length)
        
        print(f"\n✓ Sequences prepared successfully!")
        print(f"  Input shape (X): {X.shape}")
        print(f"  Output shape (y): {y.shape}")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Timesteps: {X.shape[1]}")
        print(f"  Features: {X.shape[2]}")
        
        # ============================================================================
        # STEP 4: SPLIT DATA INTO TRAIN/VALIDATION SETS
        # ============================================================================
        print("\n" + "="*80)
        print("STEP 4: Splitting Data into Train/Validation Sets")
        print("="*80)
        
        # Use 80% for training, 20% for validation
        test_size = 0.2
        random_state = 42  # For reproducibility
        
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% validation")
        print(f"Random state: {random_state} (for reproducibility)")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True  # Shuffle data before splitting
        )
        
        print(f"\n✓ Data split successfully!")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Validation samples: {X_val.shape[0]}")
        print(f"  Split ratio: {X_train.shape[0] / X.shape[0] * 100:.1f}% / {X_val.shape[0] / X.shape[0] * 100:.1f}%")
        
        # ============================================================================
        # STEP 5: BUILD LSTM MODEL
        # ============================================================================
        print("\n" + "="*80)
        print("STEP 5: Building LSTM Model")
        print("="*80)
        
        # Define input shape: (timesteps, features)
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        print(f"\nInput shape for LSTM: {input_shape}")
        print("Model architecture:")
        print("  - LSTM Layer 1: 128 units")
        print("  - Dropout: 0.2")
        print("  - LSTM Layer 2: 64 units")
        print("  - Dropout: 0.2")
        print("  - Dense Layer: 32 units (ReLU)")
        print("  - Output Layer: 1 unit (RUL prediction)")
        
        # Build the model
        model = build_lstm_model(input_shape=input_shape)
        
        print(f"\n✓ Model built successfully!")
        
        # ============================================================================
        # STEP 6: TRAIN THE MODEL
        # ============================================================================
        print("\n" + "="*80)
        print("STEP 6: Training LSTM Model")
        print("="*80)
        
        # Set training parameters
        epochs = 50
        batch_size = 32
        
        print(f"\nTraining parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: 0.001 (Adam optimizer)")
        print(f"  Loss function: MSE (Mean Squared Error)")
        
        print(f"\nTraining will include:")
        print("  - Early stopping (patience=10)")
        print("  - Model checkpointing (saves best model)")
        print("  - Learning rate reduction on plateau")
        
        # Train the model
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.0,  # Don't use internal validation since we have X_val
            verbose=1
        )
        
        print(f"\n✓ Model training complete!")
        
        # ============================================================================
        # STEP 7: SAVE THE MODEL
        # ============================================================================
        print("\n" + "="*80)
        print("STEP 7: Saving Trained Model")
        print("="*80)
        
        model_name = 'lstm_rul_predictor.h5'
        
        save_model(model, model_name=model_name)
        
        print(f"✓ Model saved successfully!")
        
        # ============================================================================
        # STEP 8: EVALUATE MODEL AND DISPLAY METRICS
        # ============================================================================
        print("\n" + "="*80)
        print("STEP 8: Evaluating Model Performance")
        print("="*80)
        
        # Evaluate on validation set
        metrics = evaluate_model(model, X_val, y_val)
        
        # ============================================================================
        # PIPELINE COMPLETE
        # ============================================================================
        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETE!")
        print("="*80)
        
        print("\nSummary:")
        print(f"  ✓ Data loaded: {raw_data.shape[0]} rows")
        print(f"  ✓ Sequences created: {X.shape[0]} sequences")
        print(f"  ✓ Model trained: {epochs} epochs")
        print(f"  ✓ Model saved: models/lstm/{model_name}")
        print(f"\nFinal Performance Metrics:")
        print(f"  - Mean Absolute Error: {metrics['mae']:.2f} cycles")
        print(f"  - Root Mean Squared Error: {metrics['rmse']:.2f} cycles")
        print(f"  - R² Score: {metrics['r2']:.4f}")
        
        print("\n" + "="*80)
        print("Next Steps:")
        print("  1. Review the training metrics above")
        print("  2. Check the saved model in models/lstm/")
        print("  3. Integrate the model into your RAG system")
        print("  4. Test predictions on new data")
        print("="*80 + "\n")
        
        return model, history, metrics
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
        print("\nPlease ensure:")
        print("  1. The C-MAPSS dataset is in data/raw/train_FD001.txt")
        print("  2. All required directories exist (run setup_folders.py)")
        traceback.print_exc()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Error during training pipeline: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the training pipeline
    try:
        model, history, metrics = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

