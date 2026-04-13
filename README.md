# Brain-Tumor-Detection

### Contributions:
Atahan Akar: Created the initial SVM model and conducted both PCA and UMAP analysis

Cyrus Carelo: Created sample notebook for SVM

Nagham Sabhour: Created sample notebook for CNN

### Introduction:
Brain tumors are among the most serious neurological conditions and early detection is critical for effective treatment. Magnetic Resonance Imaging (MRI) is widely used for identifying brain abnormalities due to its detailed imaging capabilities. However, manual inspection of MRI scans by radiologists can be time-consuming and is subject to human variability. Thus, automated tools that assist in identifying abnormal patterns in MRI images may facilitate diagnostic efficiency.

In this project, we aim to develop a machine learning-based system to classify brain MRI images according to the type of tumor present. The model will be using a unified collection of brain tumor datasets to ensure broad generalisation (https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset). The dataset is flexible for both binary (benign/malignant) and multi-classification (glioma, meningioma, and pituitary), allowing us to explore a myriad of machine learning models of varying complexity (e.g. SVM and CNN). Machine learning is appropriate for this task because tumors vary significantly in size, shape, intensity, and location. By learning patterns directly from labeled data, machine learning models may identify complex visual features that distinguish healthy tissue from abnormal regions.

Our initial approach will involve preprocessing the MRI images through resizing, normalization, and feature extraction. As a baseline model, we will train a Support Vector Machine (SVM) classifier, which is expected to perform well on high-dimensional data such as image features and can be extended to multi-class classification, making it suitable for distinguishing between different tumor categories. If time permits, we may also explore a Convolutional Neural Network (CNN) to automatically learn hierarchical image features. Model performance will be evaluated using metrics such as accuracy, precision, recall, and F1-score.
