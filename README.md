# Adaptive Illumination Normalization and Color-Contrast-Depth Segmentation Based Skin Lesion Classification Model Using Deep CNN (ALN-C2D-DCNN)

## 📘 Description
This repository presents the **ALN-C2D-DCNN** model — a robust deep learning framework for automated **skin lesion classification**.  
The model integrates **Adaptive Illumination Normalization (ALN)** and **Color-Contrast-Depth (C2D) Segmentation**, enhancing image contrast and extracting critical lesion features based on **color, contrast, and depth**.  
Using a custom **Deep Convolutional Neural Network (DCNN)**, this framework achieves up to **99% classification accuracy**, outperforming existing deep models in accuracy and efficiency.


## 🧩 Dataset Information
- **Dataset Used:** [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000), ISIC 2016-2020 challenge datasets.
- **Samples:** 10,015 dermatoscopic images.
- **Classes:** Melanoma, Nevus, Basal Cell Carcinoma, Dermatofibroma, Benign Keratosis, Actinic Keratosis, and others.
- **Metadata:** Includes patient age, gender, lesion location, and diagnostic confirmation.

---

## 💻 Code Information
### Project Structure
```
├── data/
│   ├── train/
│   ├── test/
│   └── metadata.csv
├── models/
│   └── aln_c2d_dcnn.py
├── utils/
│   ├── preprocessing.py
│   ├── segmentation.py
│   └── evaluation.py
├── requirements.txt
├── train.py
├── test.py
└── README.md
```





## Core Modules
- `preprocessing.py` – Adaptive Illumination Normalization (CIF computation)
- `segmentation.py` – C2D Segmentation and ROI extraction
- `train.py` – CNN model training with transfer learning
- `test.py` – Evaluation and classification metrics

---

## 🚀 Usage Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/researchteambts10-byte/Adaptive-Illumination-and-CCD-Skin-Lesion-Classification.git
   cd Adaptive-Illumination-and-CCD-Skin-Lesion-Classification

   
2. **Install dependencies**
    pip install -r requirements.txt

3. **Prepare dataset**
    - Download HAM10000 or ISIC dataset.
    - Place the images inside data/train/ and data/test/.
  
4. Train the model
    python train.py

5. Test and evaluate
    python test.py

6. Expected Output
    - Classification Accuracy: ~99%
    - False Classification Ratio: <3%
    - Reduced Time Complexity: ~29.7%

## ⚙️ Requirements

| Component          | Details                                                         |
| ------------------ | --------------------------------------------------------------- |
| **CPU**            | Intel Core i7-12700K @ 3.60GHz                                  |
| **GPU**            | NVIDIA GeForce RTX 3080 (10GB GDDR6X)                           |
| **RAM**            | 32 GB DDR4                                                      |
| **OS**             | Ubuntu 22.04 LTS                                                |
| **Frameworks**     | TensorFlow 2.11.0 / PyTorch 1.13.1                              |
| **CUDA Version**   | 11.7                                                            |
| **Python Version** | 3.9.13                                                          |
| **Libraries**      | NumPy, TensorFlow, PyTorch, Keras, OpenCV, Pandas, scikit-learn |

---

## 🧠 Methodology
The ALN-C2D-DCNN model operates in four major phases:

1. **Adaptive Illumination Normalization (ALN):**
   - Normalizes illumination across lesion images using neighborhood pixel intensity and red-channel based Color Intensity Factor (CIF).
   - Enhances local contrast for better lesion visibility.

2. **C2D Segmentation:**
   - Segments lesion regions using color, contrast, and depth similarity (C2DS).
   - Employs region growing to localize the Region of Interest (ROI).

3. **Transfer Learning and Data Augmentation:**
   - Applies shearing, rotation (90°, 180°, 270°), and scaling to expand dataset variability.

4. **CNN Training and Classification:**
   - Custom CNN architecture with convolution, pooling, and fully connected layers.
   - Extracted C2D features are trained and classified based on C2D Weights (C2DW).

---

## 📄 License

This project is distributed under the MIT License.



