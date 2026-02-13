# Applied-Machine-Learning-Projects

**PROJECT 1**

**Data-Driven Customer Segmentation for Targeted Marketing Using K-Means Clustering**

The goal of this project is to understand the target customer base for the marketing team to better plan their strategies. By identifying the most important customer groups based on demographic and behavioral criteria such as age, gender, income, and mall shopping score, we aim to provide actionable insights for market segmentation. This will allow the marketing team to tailor their activities and improve customer engagement and satisfaction.

**Objective**
**Market Segmentation**

The primary objective of this project is to divide the mall's target market into approachable and meaningful segments. By creating subsets of the market based on demographic and behavioral criteria, we can better understand and target specific groups for marketing activities. This process involves:

-  Demographic Segmentation: Analyzing age, gender, and income to identify key customer segments.
-  Behavioral Segmentation: Using the mall shopping score to group customers based on their shopping behavior.

**Data Collection and Preprocessing**

The first step involves collecting and preprocessing the customer data. This includes handling missing values, normalizing numerical features, and encoding categorical variables.
-  Load the raw dataset.
-  Handle missing values and outliers.
-  Normalize/standardize numerical features.
-  Encode categorical variables (e.g., gender).


**Exploratory Data Analysis (EDA)**

Exploratory Data Analysis is a crucial step to understand the underlying patterns and distributions in the data. It involves summarizing the main characteristics of the data and visualizing relationships between different variables.
-  Descriptive Statistics: Calculate mean, median, mode, standard deviation, etc., for numerical features.
-  Distribution Analysis: Visualize the distribution of age, income, and mall shopping scores.
-  Correlation Analysis: Analyze correlations between different features using heatmaps.
-  Pairwise Relationships: Use pair plots to explore relationships between multiple variables.
 
**Univariate Clustering**

Univariate clustering involves performing K-Means clustering on a single feature to identify clusters based on that feature alone.
-  Select a single feature (e.g., annual income).
-  Perform K-Means clustering.
-  Visualize the clusters using histograms or scatter plots.
-  Interpret the clusters.

**Bivariate Clustering**

Bivariate clustering involves performing K-Means clustering on two features simultaneously to explore how they interact to form distinct customer groups.
-  Select two features (e.g., annual income and mall shopping score).
-  Perform K-Means clustering.
-  Visualize the clusters using scatter plots.
-  Interpret the clusters.

**Multivariate Clustering**

Multivariate clustering involves using multiple features for clustering to capture a more comprehensive view of customer behavior and demographics.
-  Select multiple features (e.g., age, gender, annual income, mall shopping score).
-  Perform K-Means clustering.
-  Use dimensionality reduction techniques (e.g., PCA) for visualization if necessary.
-  Visualize the clusters using scatter plots or 3D plots.
-  Interpret the clusters.
 
**Visualization and Interpretation**

Visualization and interpretation of the clustering results are essential to communicate the findings effectively. This includes creating detailed visualizations and summarizing insights.
-  Create scatter plots, bar charts, and other visualizations to display the clusters.
-  Summarize the findings in a detailed report.
-  Provide actionable insights based on the identified customer segments.

**Tools**

-  Seaborn: For creating scatter plots, pair plots, and bar charts.
-  Matplotlib: For basic visualizations and customization.
-  Pandas: For data manipulation and summary statistics.
-  Sklean: For normalizing numerical features in the dataset, ensuring consistent scales for accurate clustering analysis.
  
**Results**

The clustering analysis identifies Cluster 1 as the most lucrative segment, characterized by high-income individuals with a propensity for high spending, with approximately 54% being women shoppers. Tailored marketing campaigns should focus on personalized services, exclusive offers, and loyalty programs to retain and attract these high-value customers. Additionally, targeting popular items favored by this cluster can further enhance customer engagement and drive revenue growth. Cluster 2, comprising customers with lower income and spending score, presents an opportunity for targeted sales events to stimulate spending on popular items and increase overall sales volume. By leveraging these insights, the mall can optimize marketing strategies and improve overall business performance.

---
**PROJECT 2**

**Fake News Detection**

**Objective**

The primary objective of this project is to develop a machine learning model capable of accurately detecting fake news articles. With the proliferation of misinformation and false information online, it's essential to create a reliable system that can distinguish between genuine and fabricated news content. By leveraging machine learning algorithms, the project aims to achieve accurate detection of fake news using a dataset of labeled news articles.

**Dataset**

The project utilizes two datasets:

-  Fake.csv containing fake news articles.
-  True.csv containing true news articles.

**Data Preprocessing**

The data preprocessing steps include:

-  Merging the fake and true news datasets and adding a class label (0 for fake news, 1 for true news).
-  Dropping irrelevant columns (title, subject, date) to focus on the text content and class label.
-  Shuffling the merged dataset to ensure randomness.
-  Cleaning the text data by:
      Converting text to lowercase.
      Removing special characters, URLs, and HTML tags.
      Removing punctuation and numerical values.

**Model Training**

The cleaned text data is split into training and testing sets using an 75-25 split. The TfidfVectorizer is used to convert the text data into numerical features suitable for model training.

Four machine learning models are trained on the vectorized text data:

-  Logistic Regression
-  Decision Tree Classifier
-  Gradient Boosting Classifier
-  Random Forest Classifier

**Model Evaluation**

Each model is evaluated based on accuracy, precision, recall, and F1-score. The models are compared to determine the best performer.

**Results**
The performance of each model is evaluated based on accuracy, precision, recall, and F1-score. The results demonstrate that all models perform exceptionally well in distinguishing between fake and true news:

Logistic Regression:

-  Accuracy: 98.67%
-  Precision, Recall, and F1-score: 0.99

 Decision Tree:

-  Accuracy: 99.48%
-  Precision, Recall, and F1-score: 1.00 (for the true news class)

Gradient Boosting:

-  Accuracy: 99.51%
-  Precision and Recall: High values with an F1-score of 0.99

Random Forest:


-  Accuracy: 98.96%
-  Precision, Recall, and F1-score: 0.99

Ensemble models like Gradient Boosting and Decision Tree demonstrate superior performance due to their ability to handle complex data patterns effectively.

**Manual Testing**

A manual testing function is provided to predict the class of a given news article using all four trained models. This function cleans the input text, vectorizes it using the TfidfVectorizer, and outputs the predictions from each model.

**Conclusion**

This project successfully demonstrates the use of machine learning techniques to detect fake news with high accuracy. Ensemble models, in particular, show great promise in handling the complexities of text data, making them suitable for real-world applications in fake news detection.

---

**PROJECT 3**

**Multi-Label Text Mining and Classification Using TMC2007 Dataset**

The goal of this project is to build a complete multi-label text classification pipeline capable of extracting meaningful patterns from high-dimensional unstructured text data. Unlike traditional classification tasks where each document belongs to only one category, this project focuses on multi-label classification, where each document may belong to multiple topics simultaneously.

Using the TMC2007 benchmark dataset, this project implements an end-to-end Natural Language Processing (NLP) workflow including data loading, sparse feature handling, model training, threshold tuning, and advanced evaluation metrics. The project emphasizes handling label imbalance and optimizing Macro F1-score through per-label threshold tuning.

**Objective**

The primary objective of this project is to design a scalable machine learning system that accurately assigns multiple topic labels to each document. This involves:

- Handling high-dimensional sparse text data (LIBSVM format).

- Implementing a One-vs-Rest multi-label classification strategy.

- Addressing label imbalance using Macro and Micro F1 metrics.

- Improving performance using per-label threshold tuning.

- Evaluating model robustness using multiple multi-label metrics.

**Dataset**

The project uses the TMC2007 Multi-Label Text Classification Dataset.

- Source: LIBSVM Multi-label Datasets

- Format: Sparse LIBSVM (.svm)

- Task Type: Multi-label classification

Characteristics:

- High-dimensional sparse features

- Imbalanced label distribution

- Real-world text mining benchmark dataset

Each document is represented as a sparse vector of term features, and each sample may contain multiple labels.

**Data Loading and Preprocessing**

The first step involves loading and preparing the sparse text dataset for training and evaluation.

- Load LIBSVM sparse format using Scikit-learn.

- Convert sparse matrices into appropriate multi-label format.

- Split training data into training and validation sets.

- Preserve sparse matrix structure for memory efficiency.

- Prepare label matrices for multi-label classification.

- Unlike traditional NLP pipelines, this dataset is already vectorized, allowing direct model training without TF-IDF transformation.

**Exploratory Data Analysis (EDA)**

Exploratory analysis focuses on understanding dataset characteristics rather than raw text visualization.

- Analyze number of samples and features.

- Examine label distribution across documents.

- Identify label imbalance patterns.

- Calculate average number of labels per document.

- Evaluate sparsity level of feature matrix.

Understanding label imbalance is crucial because it directly impacts Macro F1-score performance.

**Model Training**

The multi-label classification problem is handled using the One-vs-Rest strategy.

Base Model:

- Logistic Regression (Linear classifier suitable for high-dimensional sparse data)

Training Strategy:

- Train one classifier per label.

- Use sparse matrix input for computational efficiency.

- Optimize solver for high-dimensional datasets.

The model outputs probability scores for each label, which are later refined using threshold tuning.

**Threshold Optimization**

A major enhancement in this project is per-label threshold tuning.

Instead of using a fixed 0.5 threshold for all labels:

- Evaluate multiple threshold values (0.1–0.9).

- Optimize each label’s threshold individually.

- Maximize F1-score per label.

- Apply tuned thresholds to test data.

This significantly improves Macro F1-score, particularly for rare labels.

**Model Evaluation**

Since this is a multi-label classification problem, traditional accuracy is insufficient.

The model is evaluated using:

- Hamming Loss

- Micro F1-score

- Macro F1-score

Why these metrics?

- Hamming Loss measures the fraction of incorrect labels.

- Micro F1 considers label frequency (good for common labels).

- Macro F1 treats all labels equally (critical for imbalance).

  
**Results**
The model achieves strong performance on a real-world multi-label dataset:

Hamming Loss: ~0.05

Micro F1-score: ~0.68

Macro F1-score (After Threshold Tuning): ~0.65+

The improvement in Macro F1-score demonstrates effective handling of label imbalance and rare topic prediction.

Threshold tuning significantly enhances the model’s ability to detect underrepresented labels without sacrificing overall performance.

**Conclusion**

This project demonstrates a complete multi-label NLP pipeline built using production-ready machine learning practices. It highlights key challenges in text mining such as high-dimensional sparse data and label imbalance.

By implementing threshold optimization and robust evaluation metrics, the project provides a scalable and interpretable solution to real-world multi-label classification problems.

This work showcases strong foundations in:

- Natural Language Processing (NLP)

- Multi-label classification

- Model evaluation strategies

- Performance optimization

---
