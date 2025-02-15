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
│── data/              # MRI datasets (raw & processed)
│── models/            # Trained AI models
│── src/               # Code for training, evaluation, and inference
│── deployment/        # API & security implementations
│── notebooks/         # Jupyter Notebooks for experimentation
│── configs/           # Model and system configurations
│── logs/              # Training and error logs
│── requirements.txt   # Dependencies
│── README.md          # Project documentation
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
MIT License

## Contact
For inquiries, please reach out via email: `your_email@example.com`
