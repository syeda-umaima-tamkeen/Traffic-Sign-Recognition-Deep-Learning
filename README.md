#  Traffic Sign Classification Using Deep Learning

**Project by:** Syeda Umaima Tamkeen

---

## 📝 Summary

This project implements a **Traffic Sign Classification** system using **Convolutional Neural Networks (CNNs)**. The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which includes thousands of images across various traffic sign classes. The goal is to accurately classify traffic signs, a crucial task in the development of **autonomous driving systems** and **driver-assistance technologies**.

---

## 🔍 Key Steps

### 1. Dataset Loading & Preprocessing
- Loaded and visualized the GTSRB dataset containing traffic sign images across multiple classes.
- Resized images and normalized pixel values to prepare them for model training.
- One-hot encoded the class labels.

### 2. Data Augmentation
- Applied real-time augmentation (rotation, zoom, flip, shift) to improve generalization.
- Increased training data diversity to reduce overfitting.

### 3. Model Architecture (CNN)
- Built a custom CNN model with:
  - Convolutional layers
  - MaxPooling layers
  - Dropout regularization
  - Fully connected dense layers
- Used ReLU activation and softmax output for classification.

### 4. Model Training
- Compiled the model using `categorical_crossentropy` loss and `Adam` optimizer.
- Trained the model with augmented data for improved accuracy and generalization.

### 5. Evaluation
- Evaluated the trained model on test data.
- Plotted training and validation **accuracy and loss curves**.
- Generated a **confusion matrix** and classification report.

### 6. Prediction & Inference
- Tested the model on unseen images from the test set.
- Displayed predicted class vs actual class with visual plots.

---

## ✅ Results

- Achieved high accuracy on both training and test datasets.
- Successfully classified various traffic sign types even under different lighting and angle variations.
- Demonstrated robustness of CNN-based architecture for real-world image classification.

---

## 📌 Conclusion

This project showcases the effectiveness of **CNNs for visual recognition tasks**. By using a structured deep learning approach and data augmentation, the model performs well in real-world scenarios. This kind of model can be integrated into **autonomous vehicle systems** to recognize road signs and improve safety.

---

## 💡 Future Improvements

- Use of **pre-trained models** (e.g., VGG16, ResNet) for transfer learning.
- Deployment using **Streamlit** or **Flask** for real-time classification.
- Further model optimization and hyperparameter tuning.

---


## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib  
- GTSRB Dataset  
- CNN (Convolutional Neural Networks)  
- Scikit-learn (for evaluation metrics)

