# 🍎 Rotten Fruit Detection Using Deep Learning

A deep learning-based image classification model to detect **fresh** vs **rotten** fruits using **ResNet50**, built as part of a **minor project** in my undergraduate program.  
This repository contains the **machine learning pipeline** — data preprocessing, model training, evaluation, and inference.

## 📌 About the Project

The goal of this project is to build an accurate and scalable **automated fruit quality detection system** using **ResNet50**. The model can classify fruit images into `fresh` or `rotten` categories, providing real-time classification when integrated into an Android application (built separately by teammates).

This repository contains:

- ✅ Complete ML pipeline from scratch
- ✅ ResNet50 fine-tuning and training
- ✅ Data augmentation and regularization techniques
- ✅ Performance graphs and confusion matrices
- ✅ Exportable `.h5` model for deployment

## 🧠 Technologies Used

- **Python**  
- **TensorFlow / Keras**  
- **NumPy, Pandas**  
- **Matplotlib, Seaborn**  
- **ImageDataGenerator** for preprocessing  
- **ResNet50** (transfer learning)

## 🗂️ Dataset

Used two open-source Kaggle datasets, cleaned and combined:

- [Fruits Dataset by Ali Hasnain](https://www.kaggle.com/datasets/alihasnainch/fruits-dataset-for-classification)  
- [Fruits Fresh or Rotten by Sriram Reddy](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

Total: ~14,000 images across 6 fruits (Apple, Banana, Orange, Peach, Pomegranate, Strawberry), labeled as `fresh` or `rotten`.

## 📊 Model Performance

- ✅ Accuracy: ~90% on validation set
- ✅ Fine-tuned top 20 layers of ResNet50
- ✅ Used data augmentation (rotation, flip, shift, color jitter) to reduce overfitting
- 📈 Visualizations include loss/accuracy plots and confusion matrices

## 🔍 Results

- Effective classification between fresh and rotten fruit across classes
- Minor misclassifications in similar-looking classes (e.g., fresh vs rotten peaches)
- Class imbalance and limited dataset size were key challenges addressed via augmentation and dropout

## 📱 Mobile App Integration (Team Contribution)

This ML model was later integrated into a mobile app by our team as part of a collaborative minor project. The Android frontend and API backend (FastAPI) were developed jointly by teammates, while this ML pipeline was **solely developed and maintained by me**.

## 👥 Credits

- 👩‍💻 **Sugyani Krishnadarsinee** – ML development (this repository)  
- 🤝 [Rahul Saha](https://github.com/Rahulsaha30),[Srishti Singh](https://github.com/srish01ti) – Mobile app  
- 🤝 [Rohit Agarwal](https://github.com/rohitagr1) – API backend, Integration  
- 🤝 [Sreeja Upadhyaya](https://github.com/build-sreeja) – Research and UI design


## 📃 License

This project is for educational and non-commercial use only.
