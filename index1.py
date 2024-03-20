import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

# Download NLTK resources (stopwords and punkt tokenizer)
nltk.download('stopwords')
nltk.download('punkt')

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

def analyze_news(title, clf):
    # Preprocess the title
    title = preprocess_text(title)

    # Predict whether the news is real or fake
    prediction = clf.predict([title])[0]

    return prediction

if __name__ == "__main__":
    # Load the dataset
    dataset_path = input("Enter the path to the CSV file containing the dataset: ")
    df = pd.read_csv(dataset_path)

    # Display the news titles
    print(df["title"])

    # Preprocess the 'title' column
    df["title"] = df["title"].apply(preprocess_text)

    # Define 'Real' and 'Fake' classes
    df['id'] = df['id'].apply(lambda x: 'Real' if x == 0 else 'Fake')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df["title"], df["id"], test_size=0.2, random_state=42)

    # Build the Naive Bayes classifier pipeline
    clf = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the classifier
    clf.fit(X_train, y_train)

    # Get the news title from the user and analyze it
    while True:
        news_title = input("Enter a news title (or type 'exit' to quit): ")
        if news_title.lower() == 'exit':
            break
        prediction = analyze_news(news_title, clf)
        print("Prediction:", prediction)

        # Data visualization
        # Count the number of real and fake news
        real_count = df[df['id'] == 'Real'].shape[0]
        fake_count = df[df['id'] == 'Fake'].shape[0]
        # Create a pie chart
        sizes = [real_count, fake_count]
        labels = ['Real', 'Fake']
        colors = ['#66ff66', '#ff6666']
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Real vs. Fake News')
        plt.show()
