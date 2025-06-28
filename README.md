# 🧠 Brain Tumor Detection using Hybrid Deep Learning Model

This project focuses on the **automated detection and classification of brain tumors** using a **hybrid deep learning architecture** that integrates:

- **ResNet** for deep feature extraction  
- **UNet** for tumor region segmentation  
- **CNN** for tumor type classification  

The model is trained on brain MRI scans and is capable of identifying tumor presence, classifying the tumor type, and segmenting the tumor region from the image.

---

## 🚀 Key Features

- 🧼 Automatically rejects non-MRI images during preprocessing  
- 🧠 Detects the presence or absence of brain tumors  
- 🔍 Classifies tumors into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**  
- 🧬 Segments tumor regions using a UNet-based segmentation model  
- 📊 Visualizes predictions with segmentation overlays and classification results  

---

## 🧰 Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy & Pandas  
- Matplotlib & Seaborn  
- ResNet (Feature Extraction)  
- UNet (Segmentation)  
- CNN (Classification)  

---

## 🧠 Model Architecture

![Screenshot 2024-11-17 182714](https://github.com/user-attachments/assets/11dbd6fc-7bb5-4ea2-a674-fce6ba65eff2)



---

## 📦 How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/dineshthota77/Brain-Tumor-Detection-Using-Resnet-and-Unet.git
cd Brain-Tumor-Detection-Using-Resnet-and-Unet


---
##
### 2. Download the Pretrained Model

  The trained `.keras` model is not included in this repository due to GitHub’s 100 MB file size limit.

  You can manually download the model file from the link below and place it in your project directory:  

  Download Link: https://drive.google.com/file/d/1kgOYSfQgPNyYmdVP03b-WIMH5QTJJjcU/view?usp=sharing  


And put the .keras file path in app.py and run the code
---


OPTIONAL***( if you want to build your pretrained model .keras file then here is your dataset links

https://drive.google.com/drive/folders/1mt0tr8Ab6RMnlNpl9UIKYcFAyVwhrzG9?usp=sharing

so download it and in main.py update the paths)***OPTIONAL



---

Output: 



![Screenshot (52)](https://github.com/user-attachments/assets/9153238f-3d8d-49ec-9a57-217942731ae8)
![Screenshot (55)](https://github.com/user-attachments/assets/b6156157-7796-40e9-8da5-6ebcc50c0953)
![Screenshot (55)](https://github.com/user-attachments/assets/0ffd1905-12b5-4f9a-ac07-3af475ba8e3d)
![Screenshot (54)](https://github.com/user-attachments/assets/3cd51857-5c17-4357-94db-1ccc41045f8c)


---
