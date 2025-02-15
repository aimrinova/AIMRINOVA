# AIMRINOVA: AI-Powered MRI Disease Detection

## Overview
AIMRINOVA is an AI-powered system designed for on-premise deployment in hospitals and radiology centers to assist in MRI scan analysis. It leverages deep learning models to classify and detect diseases from MRI scans, ensuring privacy and security by processing data locally on hospital networks.

## Features
- **On-Premise Deployment**: Runs locally on NVIDIA Digits PC or similar hardware.
- **Deep Learning-Based Detection**: Uses CNN-based models (ResNet, U-Net, EfficientNet) for accurate disease classification.
- **DICOM Image Processing**: Converts and preprocesses MRI scans for analysis.
- **Automated Diagnosis**: AI-based inference to highlight potential abnormalities.
- **Security & Compliance**: HIPAA and GDPR-compliant data encryption and patient privacy.

## Project Structure
```
AIMRINOVA/          
│── /data                          # MRI datasets (raw & processed)
│   │── /raw_mri_scans             # Original DICOM/NIfTI scans
│   │── /processed_mri_scans       # Preprocessed MRI images
│   │── /segmentation_masks        # Ground truth masks for U-Net
│   │── /predictions               # Model output (classified images, segmented heatmaps)
│
│── src/               # Code for training, evaluation, and inference
│── /models                        # Trained AI models
│   │── mri_model_v1.h5            # CNN model (initial version)
│   │── mri_model_v2.h5            # Improved model
│   │── unet_segmentation.h5       # U-Net segmentation model
│
│── /scripts
│   │── preprocess_mri.py          # Preprocess MRI images (resize, normalize, denoise)
│   │── train_cnn.py               # Train CNN model for classification
│   │── train_unet.py              # Train U-Net model for segmentation
│   │── predict_mri.py             # Load trained models and predict injuries
│   │── grad_cam.py                # Generate heatmaps for explainability
│   │── evaluate_model.py          # Compute accuracy, precision, recall, confusion matrix
│
│── /deployment                    # API & security implementations
│   │── flask_api.py               # REST API for model inference (for local hospital use)
│   │── gui_app.py                 # Optional: Local GUI for MRI uploads
│   │── /configs
│   │   │── model_config.json      # Configuration for loading models
│   │   │── requirements.txt       # Dependency list
│
│── /logs
│   │── training_logs.txt          # Logs from model training
│   │── inference_logs.txt         # Logs from model predictions
│
│── notebooks/         # Jupyter Notebooks for experimentation
│── configs/           # Model and system configurations
│── requirements.txt   # Dependencies
│── setup_project.sh                # Project setup script (creates folders, installs dependencies)
│── README.md                        # Project documentation
│── .gitignore   
```

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/AIMRINOVA.git
cd AIMRINOVA
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run Preprocessing Script**
```bash
python src/preprocessing.py --input data/raw --output data/processed
```

### **4. Train the AI Model**
```bash
python src/train.py --config configs/model_config.yaml
```

### **5. Evaluate the Model**
```bash
python src/evaluate.py --model models/best_model.pth --test data/test
```

### **6. Run Inference on New MRI Scans**
```bash
python src/infer.py --input scans/mri_sample.dcm
```

## Deployment
- **On-Premise**: Deployed as a local API using Flask/FastAPI.
- **Hospital Integration**: Compatible with PACS/DICOM systems.
- **Security**: Implements encryption to separate patient data from images.

## Future Enhancements
- Support for 3D MRI scans.
- Real-time analysis using NVIDIA Triton Inference Server.
- Expansion to other imaging modalities (CT, X-ray).

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Create a Pull Request.

## License
Proprietary License

## Contact
For inquiries, please reach out via email: `daschub496@gmail.com`
