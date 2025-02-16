# AIMRINOVA

### **📌 AIMRINOVA – AI-Powered MRI Scan Analysis**
🚀 **Revolutionizing Medical Imaging with On-Premises AI for MRI Disease Detection**  

---

## **📖 Overview**
AIMRINOVA is an AI-powered MRI scan analysis system designed for hospitals and radiologists. It leverages deep learning techniques, particularly **Convolutional Neural Networks (CNNs)**, to assist in **detecting abnormalities in MRI scans** with high accuracy.  

Unlike cloud-based solutions, AIMRINOVA runs **on-premises** using **Nvidia Project Digits or custom AI hardware**, ensuring **data privacy, HIPAA/GDPR compliance, and reduced processing costs**.

---

## **💡 Features**
✔ **On-Premises AI Processing** – No cloud dependency, ensuring full patient data security.  
✔ **Deep Learning for MRI Scans** – Uses CNN models trained on medical imaging datasets.  
✔ **Customizable AI Models** – Supports model fine-tuning and continuous learning.  
✔ **Fast & Automated Diagnosis** – Reduces radiologists’ workload by pre-screening scans.  
✔ **Secure Data Handling** – Image processing is separated from patient data for added privacy.  
✔ **Plug-and-Play Deployment** – Compatible with **Nvidia Digits, custom servers, and hospital PACS systems**.

---

## **📂 Project Structure**
```
AIMRINOVA/
│── data/                          # MRI data storage
│   │── brain_tumor/               # Brain tumor MRI dataset
│   │   │── raw_mri_scans/         # Original DICOM/NIfTI scans
│   │   │── processed_mri_scans/   # Preprocessed images
│   │   │── segmentation_masks/    # Ground truth segmentation labels
│   │   │── predictions/           # AI model predictions (heatmaps, classifications)
│   │── spinal_injury/             # Spinal MRI dataset
│   │── dental_mri/                # Dental MRI dataset for dentists
│   │── other_diseases/            # Future expansions
│
│── models/                        # Trained AI models per disease
│   │── brain_tumor/
│   │   │── cnn_brain_tumor.engine  # TensorRT-optimized classification model
│   │   │── unet_brain_tumor.engine # Segmentation model
│   │── spinal_injury/
│   │── dental_mri/
│   │── other_diseases/
│
│── src/                           # Core AI code for training, inference, and evaluation
│   │── training/                   # Disease-specific training scripts
│   │   │── train_brain_tumor.py
│   │   │── train_spinal_injury.py
│   │   │── train_dental.py
│   │   │── convert_tensorrt.py     # Converts trained models to TensorRT format
│   │── inference/                  # Inference & model predictions
│   │   │── predict_mri.py
│   │   │── grad_cam.py             # Explainability (heatmaps)
│
│── scripts/                        # Utility scripts
│   │── preprocess_mri.py           # Resize, normalize, denoise MRI images
│   │── evaluate_model.py           # Accuracy, precision, recall, confusion matrix
│
│── deployment/                     # API & GUI for hospitals
│   │── api/                         # Model inference API
│   │   │── django_api/              # Django-based REST API (for hospitals)
│   │   │── flask_api.py             # (Alternative) Flask-based API
│   │── gui/                         # Local GUI for MRI uploads (if needed)
│   │── docker/                      # Dockerized deployment setup
│   │   │── Dockerfile
│   │   │── tensorrt_runtime.sh
│
│── config/                          # Model configuration files
│   │── diseases/
│   │   │── brain_tumor.json
│   │   │── spinal_injury.json
│   │   │── dental_mri.json
│
│── logs/                            # Training & inference logs
│   │── training_logs.txt
│   │── inference_logs.txt
│   │── dashboard_logs/              # Logs for the web-based dashboard
│
│── dashboard/                       # Django-based training dashboard
│   │── aimrinova_dashboard/         # Django project root
│   │   │── manage.py                # Django project manager
│   │   │── dashboard_app/           # Django app for dashboard
│   │   │   │── models.py            # Training logs, user roles
│   │   │   │── views.py             # API endpoints for real-time updates
│   │   │   │── urls.py              # URL routing
│   │   │   │── consumers.py         # WebSockets for real-time updates
│   │   │   │── templates/           # Web UI (HTML)
│   │   │   │── static/              # CSS, JS, images
│   │   │── users/                   # User authentication & role management
│   │   │   │── models.py            # User roles (radiologists, IT admins)
│   │   │   │── views.py             # Login, registration
│
│── ui/                              # Web interface for radiologists
│── docs/                            # Documentation & compliance  
│── tests/                           # Unit & performance tests  
│── LICENSE                          # Proprietary License  
│── README.md                        # Project documentation  
```

---
### **Features**
✅ **Django Dashboard** 📊 – A dedicated app for monitoring training progress.  
✅ **Multi-User Support** 👤 – Role-based access for radiologists & IT admins.  
✅ **WebSockets** ⚡ – Real-time updates without needing manual refresh.  
✅ **Ethernet/LAN Support** 🌐 – Hospitals can use the dashboard **locally**.  

---

## **⚙️ Tech Stack**
🔹 **AI Frameworks**: TensorFlow, PyTorch, MONAI (Medical AI Toolkit)  
🔹 **Data Format**: DICOM (Digital Imaging and Communications in Medicine)  
🔹 **Processing Hardware**: Nvidia **Project Digits** or **Custom AI Server**  
🔹 **Networking & Security**: Encrypted data processing for **GDPR/HIPAA compliance**  
🔹 **Hospital Integration**: PACS (Picture Archiving and Communication System), HL7  

---

## **🛠️ Installation**
### **1️⃣ Setup the Environment**
```bash
git clone https://github.com/yourusername/AIMRINOVA.git
cd AIMRINOVA
pip install -r requirements.txt
```
### **2️⃣ Prepare the Data**
Ensure your MRI scans are in **DICOM format** and placed inside `data/`. Run preprocessing:
```bash
python scripts/preprocess.py --input data/ --output processed_data/
```

### **3️⃣ Train the AI Model**
```bash
python scripts/train.py --config config/train_config.yaml
```

### **4️⃣ Deploy & Run Inference**
```bash
python src/inference.py --image data/sample_mri.dcm
```

---

## **🔬 Model Training & Evaluation**
### **🧠 Training CNN Model**
AIMRINOVA uses **Convolutional Neural Networks (CNNs)** trained on MRI scan datasets:
- **Pretrained Networks**: EfficientNet, ResNet, UNet for segmentation.
- **Optimization**: Uses **cross-validation**, **hyperparameter tuning**, and **data augmentation**.
- **Evaluation Metrics**: Accuracy, Sensitivity, Specificity, Dice Score, F1-score.

### **📊 Continuous Improvement**
- Uses **active learning** to enhance detection accuracy.
- Supports **federated learning** (optional) for multi-hospital AI model improvements.

---

## **🔐 Security & Compliance**
✔ **Patient Data Encryption** – MRI scans are processed **without personal data exposure**.  
✔ **Compliant with GDPR & HIPAA** – No cloud uploads; all AI processing is local.  
✔ **Data Integrity & Audit Logs** – Ensures all AI results can be reviewed by medical professionals.  

---

## **🚀 Roadmap**
- [x] Build AI model for MRI scan classification  
- [x] Develop hospital integration (DICOM, PACS, HL7)  
- [ ] Implement user-friendly **radiologist UI**  
- [ ] Validate with real hospital test cases  
- [ ] Optimize AI model for **real-time inference**  

---

## **📜 License**
🔒 **Proprietary License** – AIMRINOVA is **not open-source**. All rights reserved.  

For commercial use or licensing inquiries, **contact us**.  

---

## **📩 Contact**
💼 **Project Lead**: *Daniel Schubert*  
📧 **Email**: *daschub496@gmail.com*  
🌐 **Website**: *[AIMRINOVA Homepage](https://www.aimrinova.com/)*
🌐 **Techstack**: *[AIMRINOVA GitHub Repository](https://github.com/aimrinova/AIMRINOVA/)*

---
