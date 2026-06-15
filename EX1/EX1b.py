import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Read the dataset
df = pd.read_csv(r"D:\SEM_5\ML\datasets\Text_data\IMDB Dataset.csv")
reviews = df['review']
text = " ".join(reviews.astype(str))

# 2. Count the number of sentences
sentences = sent_tokenize(text)
print("\nNumber of Sentences:")
print(len(sentences))

# 3. Count the number of words
words = word_tokenize(text)
words = [word.lower() for word in words if word.isalpha()]
print("\nNumber of Words:")
print(len(words))

# 4. Find the 10 most frequent words
word_freq = Counter(words)
print("\nTop 10 Most Frequent Words:")
for word, count in word_freq.most_common(10):
    print(f"{word} : {count}")

# 5. Remove stop words and compare results
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words
                  if word not in stop_words]
filtered_freq = Counter(filtered_words)
print("\nTop 10 Words After Removing Stop Words:")
for word, count in filtered_freq.most_common(10):
    print(f"{word} : {count}")

# 6. Calculate Vocabulary Size
vocabulary = set(filtered_words)
print("\nVocabulary Size:")
print(len(vocabulary))

# 7. Find Average Sentence Length
total_words = len(words)
total_sentences = len(sentences)
avg_sentence_length = total_words / total_sentences
print("\nAverage Sentence Length:")
print(round(avg_sentence_length, 2), "words")

# 8. Generate Word Cloud
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(" ".join(filtered_words))
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of IMDb Reviews")
plt.show()