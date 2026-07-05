import pandas as pd 
from textblob import TextBlob
import matplotlib.pyplot as plt
 
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load data set----
df= pd.read_csv("data/amazon_reviews.csv")

# Text preprocessing Function:-------------

def clean_text(text):

    text= text.lower()
    text=re.sub(r'[^a-zA-Z\s]', '', text)

    stop_words= set(stopwords.words('english'))

    words = text.split()

    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply text cleaning----------------

df["Cleaned_Review"]= df["Review"].apply(clean_text)


# Sentiment Analysis Function-------------

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity>0:
        return "Positive"
    elif polarity<0:
        return "Negative"
    else:
        return"Neutral"
    
# apply sentiment analysis--

df["Sentiment"]= df["Review"].apply(get_sentiment)

print(df.head())

print("\nSentiment Count:")

print(df["Sentiment"].value_counts())


# Bar Chart------
sentiment_count= df["Sentiment"].value_counts()

sentiment_percentage= (sentiment_count/sentiment_count.sum())*100
print("\n Sentiment Percentage")

print(sentiment_percentage.round(2))

plt.figure(figsize= (6,4))

sentiment_count.plot(kind="bar")

plt.title("Sentiment Analysis of Amazon Reviews ")
plt.xlabel("Sentiment")

plt.ylabel("Number of Reviews")
plt.savefig("bar_chart.png")
plt.show()

# Pie Chart------------

plt.figure(figsize=(6,4))

# sentiment_count.plot( kind="pie",autopic= "%1.1f%%",startangle=90)
plt.pie(sentiment_count, labels= sentiment_count.index,autopct="%1.1f%%", startangle=90)
plt.title("Sentiment distribution")
plt.ylabel("")
plt.savefig("pie_chart.png")
plt.show()

df.to_csv("sentiment_output.csv", index= False)

print("Output file saved successfully")

#Business Insight----------

print("\Bussiness Insight:")

if sentiment_count["Positive"]>sentiment_count["Negative"]:
    print("Most custumer are satisfied with the product")
    print("The product have positive market response.")
    print("The company should maintain quality and improve existance feature.")

else:
    print("Many customers are dissatisfied with the product.")
    print("The company should improve product quality ans customer service.")

# WORD CLOUD-----------

# from worldcloud import WordCloud

# text= " ".join(df["Cleaned_Review"])

# # create World Cloud--

# wordcloud = WordCloud(
#     width =800,
#     height=400,
#     background_color="white"
# ).generate(text)

# #display Word Cloud

# plt.figure(figsize=(10,5))
# plt.imshow(wordcloud,interpolation="billinear")
# plt.axis("off")
# plt.title("Word Cloud of Amazon Reviews ")

# plt.show()

# df["Cleaned_Review"]= df["Review"].apply(clean_text)

# print(df[["Review","Cleaned_Review"]])

# df= pd.read_csv("data/amazon_reviews.csv")

# df["Cleaned_Review"]= df["Review"].apply(clean_text)


# print("\nSentiment Percentage")
# sentiment_percentage= (df["Sentiment"].value_counts(normalize=True)*100).round(2)
