import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

# Download NLTK resources (stopwords and punkt tokenizer)
nltk.download('stopwords')
nltk.download('punkt')

# Set the CSV file paths
gossipcop_fake_path = 'FAKE NEWS.PY/gossipcop_fake.csv'
gossipcop_real_path = 'FAKE NEWS.PY/gossipcop_real.csv'
politicalfact_fake_path = 'FAKE NEWS.PY/politifact_fake.csv'
politicalfact_real_path = 'FAKE NEWS.PY/politifact_real.csv'

# Load the CSV files
gossipcop_fake_df = pd.read_csv(gossipcop_fake_path, encoding='utf-8')
gossipcop_real_df = pd.read_csv(gossipcop_real_path, encoding='utf-8')
politicalfact_fake_df = pd.read_csv(politicalfact_fake_path, encoding='utf-8')
politicalfact_real_df = pd.read_csv(politicalfact_real_path, encoding='utf-8')

# Combine the dataframes
df = pd.concat([gossipcop_fake_df, gossipcop_real_df, politicalfact_fake_df, politicalfact_real_df], ignore_index=True)

# Print the columns of the dataset
print("Columns of the dataset:")
print(df.columns)

# Print the first few rows of the loaded dataset
print("Loaded Dataset:")
print(gossipcop_fake_df.head())
print(gossipcop_real_df.head())
print(politicalfact_fake_df.head())
print(politicalfact_fake_df.head())

# Preprocessing
def preprocess_text(text):
    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Join words back into a single string
    return " ".join(words)

# Apply preprocessing to the 'title' column
df["title"] = df["title"].apply(preprocess_text)

# Define 'Real' and 'Fake' classes
df['id'] = df['id'].apply(lambda x: 'Real' if x == 0 else 'Fake')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["title"], df["id"], test_size=0.2, random_state=42)

# Build the Naive Bayes classifier pipeline
clf = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix_result)
print("Classification Report:\n", classification_report_result)

# Create data for the pie chart
labels = ['Real', 'Fake']
real_count = df['id'].value_counts().get('Real', 0)
fake_count = df['id'].value_counts().get('Fake', 0)
sizes_pie = [real_count, fake_count]

# Create data for the bar graph
sizes_bar = [real_count, fake_count]
categories = ['Real', 'Fake']

# Create a pie chart with decorations
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Pie chart
colors = ['#66ff66', '#ff6666']  # Green for Real, Red for Fake
explode = (0.1, 0)  # explode the 'Real' slice slightly
ax[0].pie(sizes_pie, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode, shadow=True,
          wedgeprops=dict(width=0.3))
ax[0].set_title('Fake News Detection - Pie Chart')

# Bar graph
ax[1].bar(categories, sizes_bar, color=['#66ff66', '#ff6666'])
ax[1].set_title('Fake News Detection - Bar Graph')

# Display the charts
plt.show()
