# Spam-Email-Detection-Using-Naive-Bayes-Classifier

This project implements a spam email detection system using natural language processing and machine learning techniques. It utilizes a Naive Bayes classifier to classify emails as either spam or ham (non-spam) based on their content.

Prerequisites
To run this project, you need to have the following software installed:
	Python 3.x
	Google Colab (for cloud-based execution)
  
	Libraries:
	pandas
	numpy
	re
	nltk
	scikit-learn

Dataset
The dataset used for this project is spam.csv, which contains labeled email messages categorized as spam or ham. The kaggle link for the same is: https://www.kaggle.com/code/mfaisalqureshi/email-spam-detection-98-accuracy/input

You can upload the dataset in your Google Drive. Ensure that the dataset is in the following directory structure if you want to run the code as it is, or define a new directory of where you have chosen to upload the dataset:
/content/drive/MyDrive/spam.csv

Accessing Google Drive
To access the dataset stored in your Google Drive while using Google Colab, you need to mount your Google Drive as follows:
Import the drive module from Google Colab:
	from google.colab import drive
Mount your Google Drive:
	drive.mount('/content/drive')
Specify the file path to the dataset:
	file_path = '/content/drive/MyDrive/spam.csv'

Project Structure
Data Importing: The project begins by importing the necessary libraries and loading the dataset from Google Drive.

Data Preprocessing: The text data is cleaned and prepared for analysis. This involves:
Converting text to lowercase
Removing special characters and numbers
Stemming the words
Removing common stopwords

Model Training:
The cleaned text data is split into features (X) and labels (y).
The dataset is further divided into training and testing sets.
A TF-IDF vectorizer is used to convert the text data into numerical feature vectors.
A Multinomial Naive Bayes classifier is trained on the training data.

Model Evaluation:
The model's accuracy is evaluated on the test data.
A classification report and confusion matrix are displayed for detailed performance metrics.
Hyperparameter Tuning: The model is further improved using GridSearchCV to find the best hyperparameter values.

Execution Steps:
Open Google Colab.
Copy and paste the project code into a new notebook.
Ensure the spam.csv file is correctly placed in your Google Drive.
Run each code cell sequentially to execute the project.

Results:
The project outputs the following metrics:
Accuracy of the model on the test data.
Classification report indicating precision, recall, and F1-score for each class.
Confusion matrix showing the number of correct and incorrect predictions.

Conclusion
This project demonstrates the implementation of a spam email detection system using machine learning techniques, achieving high accuracy rates through data preprocessing, model training, and hyperparameter tuning.
