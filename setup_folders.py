import os

def create_folder_structure():
    """Create the project folder structure for RAG Predictive Maintenance."""
    
    # Define the folder structure
    folders = [
        # Data folders
        'data/raw',
        'data/processed',
        'data/test',
        
        # Model folders
        'models/lstm',
        'models/rag',
        
        # Maintenance manuals
        'maintenance_manuals/pdfs',
        
        # Source code folders
        'src/data_preprocessing',
        'src/model_training',
        'src/rag_system',
        'src/utils',
        
        # Notebooks
        'notebooks',
        
        # Results folders
        'results/figures',
        'results/logs',
        
        # App folder
        'app',
    ]
    
    print("Creating folder structure...")
    
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"✓ Created: {folder}")
        except Exception as e:
            print(f"✗ Error creating {folder}: {e}")
    
    print("\nFolder structure created successfully!")

if __name__ == "__main__":
    create_folder_structure()

