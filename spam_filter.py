# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# Create Data class that loads and splits data
class Data:
  def __init__(self):
    self.df = pd.read_csv('sample_emails.csv')
    self.random_df = pd.read_csv('50_random_emails.csv')
  
  # Sample email data for training and testing
  def split_sample_data(self):
    X_train, X_test, y_train, y_test = train_test_split(self.df['text'], self.df['spam'], test_size=0.2, random_state=42)
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test

  # 50 Random Purdue email data for testing
  def split_random_data(self):
    self.X_random = self.random_df['text']
    self.y_random = self.random_df['spam']

# Initialize Data class
dataset = Data()
dataset.split_sample_data()
dataset.split_random_data()

# View dataset head
dataset.df.head()

# View dataset information
dataset.df.info()

# View shape of dataset
print(f"Shape of Sample Dataset: [{dataset.df.shape[0]}, {dataset.df.shape[1]}]")
print(f"Shape of Random Dataset: [{dataset.random_df.shape[0]}, {dataset.random_df.shape[1]}]")

def calculate_hinge_loss(y_true, y_pred):
  y_replace_0 = y_true.replace(0, -1) # replace 0 to -1 (class of ham emails becomes -1)
  product = y_replace_0 * y_pred
  hinge_losses = np.maximum(0, (1 - product))
  hinge_losses = np.mean(hinge_losses)
  return hinge_losses

def train_svm_model(dataset, epochs):
  # Vectorize X data for training
  vectorizer = CountVectorizer(stop_words='english') # remove English stop words
  X_train_vector = vectorizer.fit_transform(dataset.X_train)

  # Initialize SGD-based SVM model
  sgd = SGDClassifier(loss='hinge', max_iter=1, warm_start=True, random_state=42)
  classes = [0, 1] # 0: ham, 1: spam

  # Repeat training and print hinge loss
  for epoch in range(epochs):
      sgd.partial_fit(X_train_vector, dataset.y_train, classes=classes)

      # Calculate hinge loss
      y_pred = sgd.decision_function(X_train_vector)
      hinge_losses = calculate_hinge_loss(dataset.y_train, y_pred)

      print(f"Epoch {epoch+1}/{epochs}, Hinge Loss: {hinge_losses:.4f}")

  return sgd, vectorizer

# Initialize and train the model using n epochs
n = 10
sgd, vectorizer = train_svm_model(dataset, n)

def plot_frequent_words(dataset, n, email_type):
    # Remove word "Subject" (All texts contain it)
    texts = dataset.df[dataset.df['spam'] == email_type]['text']
    texts = texts.str.replace(r"(?i)subject:\s*", "", regex=True)

    # Vectorize texts
    vectorizer = CountVectorizer(stop_words='english')
    text_vector = vectorizer.fit_transform(texts)

    # Count the words
    counts = text_vector.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    # Create table in descending order
    result = pd.Series(counts, index=words)
    result = result.sort_values(ascending=False).head(n)

    # Plot a bar graph
    plt.figure(figsize=(5, 5))
    sns.barplot(x=result.values, y=result.index)
    if email_type == 1:
      plt.title(f"Top {n} Most Frequent Words in Spam Emails (Excluding 'Subject')")
    else:
      plt.title(f"Top {n} Most Frequent Words in Ham Emails (Excluding 'Subject')")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.show()

n = 20
# Visualize n most frequent spam words
plot_frequent_words(dataset, n, 1)

# Visualize n most frequent ham words
print("\n")
plot_frequent_words(dataset, n, 0)

def plot_confusion_matrix(dataset, model, vectorizer):
  # Vectorize test data and predcit email type
  X_test_vector = vectorizer.transform(dataset.X_test)
  y_pred = model.predict(X_test_vector)

  # Create confusion matrix
  cm = confusion_matrix(dataset.y_test, y_pred)
  cm_label = pd.DataFrame(cm, index=['Actual Ham', 'Actual Spam'], columns=['Predicted Ham', 'Predicted Spam'])

  # Plot confusion matrix
  plt.figure(figsize=(5, 5))
  sns.heatmap(cm_label, annot=True, cmap='Greens')
  plt.title("Confusion Matrix")
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.show()

  # Display accuracy score
  accuracy = accuracy_score(dataset.y_test, y_pred) * 100
  print(f"\nAccuracy Score: {accuracy:.2f}%")

plot_confusion_matrix(dataset, sgd, vectorizer)

def plot_roc_curve(dataset, model, vectorizer):
  # Vectorize X data for test
  X_test_vector = vectorizer.transform(dataset.X_test)

  # Predict email type of vectorized X data
  y_decision_score = model.decision_function(X_test_vector)

  # Calculate ROC and AUC
  fprs_test, tprs_test, _ = roc_curve(dataset.y_test, y_decision_score)
  auc_test = auc(fprs_test, tprs_test)

  # Plot ROC curves
  plt.figure(figsize=(5, 5))
  plt.plot(fprs_test, tprs_test, label=f'Test ROC (AUC = {auc_test:.4f})', linestyle='-', color='blue')
  plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', color='red')
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Curve")
  plt.legend(loc="lower right")
  plt.grid(True)
  plt.show()

plot_roc_curve(dataset, sgd, vectorizer)

def test_random_purdue_emails(dataset, model, vectorizer):
  # Vectorize and predict random purdue email data
  X_test_vector = vectorizer.transform(dataset.X_random)
  y_decision_score = model.decision_function(X_test_vector)
  predicted = model.predict(X_test_vector)

  # Create table containing results of each email
  results = pd.DataFrame({'score': y_decision_score, 'correct': predicted == dataset.y_random})

  # Plot a graph
  colors = results['correct'].map({True: 'green', False: 'red'})
  plt.figure(figsize=(12, 5))
  plt.bar(range(len(results)), results['score'], color=colors)
  plt.title("Desicion Scores of 50 Random Purdue Emails (Green = Correct, Red = Incorrect)")
  plt.xlabel("Emails")
  plt.ylabel("Decision Score")
  plt.tight_layout()
  plt.show()

test_random_purdue_emails(dataset, sgd, vectorizer)

def display_email(email_text, email_type, classification_result):
    text = "\n".join(textwrap.wrap(email_text, width=55))

    if (email_type == 1):
      text = "<Acutal Purdue Spam Email>\n\n" + text
    else:
      text = "<Acutal Purdue Ham Email>\n\n" + text

    text = text + "\n\n-> " + classification_result

    plt.figure(figsize=(5, 1))
    plt.axis('off')
    plt.text(0.05, 0.5, text, fontsize=12)
    plt.show()

def sigmoid(decision_score):
  return 1 / (1 + np.exp(-decision_score))

def classify_email(email_text, email_type, model, vectorizer):
  # Vectorize and predict type of received email
  text_vector = vectorizer.transform([email_text])
  decision_score = model.decision_function(text_vector)[0]
  prediction = model.predict(text_vector)[0]

  # Calculate correctness probability using sigmoid
  proba_spam = sigmoid(decision_score)
  proba_ham = 1 - proba_spam

  if (prediction == 0):
    label = "Ham"
    prob = proba_ham * 100
  else:
    label = "Spam"
    prob = proba_spam * 100

  classification_result = f"The email was classified as {label} with {prob:.2f}% probability."
  display_email(email_text, email_type, classification_result)

# Test each spam and ham email sent to Purdue email account
with open("purdue_spam_email.txt", "r", encoding="utf-8") as f:
  purdue_spam_email = f.read()
  classify_email(purdue_spam_email, 1, sgd, vectorizer)
  print("\n")
  
with open("purdue_ham_email.txt", "r", encoding="utf-8") as f:
  purdue_ham_email = f.read()
  classify_email(purdue_ham_email, 0, sgd, vectorizer)