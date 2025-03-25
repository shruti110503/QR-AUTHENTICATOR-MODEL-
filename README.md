# QR CODE AUTHENTICATOR (MODEL)

## **Project Overview**
QR Code Authenticator is a machine learning-based system designed to distinguish between **original (first print) QR codes** and **counterfeit (reprinted) QR codes**. The project utilizes **Support Vector Machines (SVM)** and **Convolutional Neural Networks (CNN - PyTorch)** to achieve high-accuracy classification.

## **Features**
✅ Identifies original vs. reprinted QR codes with **100% accuracy** on the given dataset.  
✅ Implements **two classification models**: SVM (traditional ML) and CNN (deep learning).  
✅ Works with **grayscale QR code images** resized to **128x128 pixels**.  
✅ Can be extended for **real-world deployment** via mobile apps, cloud APIs, or embedded systems.  

## **Dataset**
The dataset consists of **200 QR code images**, equally split into:
- **First Prints (Original QR Codes) – 100 images**
- **Second Prints (Counterfeit QR Codes) – 100 images**

Each image has subtle variations in print quality, texture, and microscopic patterns.

## **Technologies Used**
🔹 **Python** (OpenCV, NumPy, Matplotlib)  
🔹 **Machine Learning** (Scikit-Learn - SVM)  
🔹 **Deep Learning** (PyTorch - CNN)  
🔹 **Data Handling** (Pandas, OpenCV)  

## **Installation & Setup**
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/qr-code-authenticator.git
cd qr-code-authenticator
```

### **Step 2: Install Dependencies**
```bash
pip install numpy opencv-python matplotlib scikit-learn torch torchvision
```

### **Step 3: Set Up Dataset**
- Extract the dataset folders (`First_Print`, `Second_Print`) into the project directory.
- Update the paths in the script (`first_print_path`, `second_print_path`).
- https://drive.google.com/drive/folders/1pPeWT1zntlKXnuY_yHmpI-ZzKl4nLgQS?usp=drive_link (Data Set Link)

### **Step 4: Run the Model**
```bash
python qr_authenticator.py
```

## **Results & Evaluation**
Both models achieved **100% accuracy** on the given dataset:
| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|-----------|-----------|
| **SVM** | 1.00 | 1.00 | 1.00 | 100% |
| **CNN (PyTorch)** | 1.00 | 1.00 | 1.00 | 100% |

## **Deployment Possibilities**
Your trained model can be deployed in **real-world scenarios** like:
- **Mobile App (Android/iOS)** – Scan QR codes and verify authenticity in real-time.
- **Cloud API** – Host the model as a REST API for businesses to integrate.
- **Embedded System** – Deploy on a Raspberry Pi or edge device for hardware authentication.

## **Next Steps**
🚀 **Planned Improvements:**
- Data Augmentation (to improve generalization)
- Real-world QR Code Testing
- Mobile App Integration

## **License**
This project is licensed under the **MIT License**.

---
### **Need Help?**
If you have any questions, feel free to open an issue or reach out!

