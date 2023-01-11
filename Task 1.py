#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("BA_reviews.csv")
df.head()


# In[3]:


##Exploratory Data Analysis


# In[4]:


df.drop("Unnamed: 0", axis=1,inplace=True)


# In[5]:


df.head()


# In[6]:


for i in range(0, len(df)):
    try:
        df.loc[i, 'reviews'] = df.loc[i, 'reviews'].split('|')[1]
    except:
        pass


# In[7]:


df.isnull().sum()


# In[8]:


len(df)


# In[9]:


##Data Cleaning


# In[10]:


import re

# Define a function to clean the text
def clean(text):
# Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', str(text))
    return text
# Cleaning the text in the review column
df['Reviews'] = df['reviews'].apply(clean)
df.head()


# In[11]:


df.drop("reviews", axis=1,inplace=True)


# In[12]:


df["Reviews"].str.lower()


# In[13]:


#Applying the preprocessing on the dataset


# In[14]:


df["Reviews"][0:10]


# In[15]:


#Tokenization
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('wordnet')

def tokenize_words(text):
    # Tokenize the text by words
    tokens = word_tokenize(text)
    # Return the tokens
    return tokens

def tokenize_sentences(text):
    # Tokenize the text by sentences
    tokens = sent_tokenize(text)
    # Return the tokens
    return tokens


# In[16]:


df["Reviews"] = df["Reviews"].apply(tokenize_words)


# In[17]:


from nltk import pos_tag
from nltk.corpus import wordnet as wn

def pos_tag_text(text):
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    tagged_text = pos_tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text

df['Reviews'] = df['Reviews'].apply(pos_tag_text)
df.head()


# In[18]:


#Stemming
#from nltk.stem import PorterStemmer

#def stem_words(tokens):
    # Create a PorterStemmer object
    #stemmer = PorterStemmer()
    # Stem the tokens
    #stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Return the stemmed tokens
    #return stemmed_tokens


# In[19]:


#df["Reviews"] = df['Reviews'].apply(stem_words)


# In[20]:


#Lemmatizer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize_words(pos_data):
    l_row = " "
    for word, pos in pos_data:
     if not pos:
        lemma = word
        l_row = l_row + " " + lemma
     else:
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        l_row = l_row + " " + lemma
    return l_row

df['Reviews'] = df['Reviews'].apply(lemmatize_words)
df.head()


# In[21]:


#for index, row in df.iterrows():
#    filter_sentence = []
#   sentence = row["Cleaned Reviews"]
#    sentence = re.sub(r'[^\w\s]','',sentence) #Cleaning
#    words = nltk.word_tokenize(sentence) #tokenization
#    sentence = [w for w in words if not w in stop_words] #stopwords removal
#    for word in words:
#        filter_sentence.append(lemmatizer.lemmatize(word))
#    print(filter_sentence)
#   df.loc[index,"Cleaned Reviews"] = filter_sentence


# In[22]:


len(df["Reviews"])


# In[23]:


##Sentiment Analysis using VADER
get_ipython().system('pip install vaderSentiment')


# In[24]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# function to calculate vader sentiment
def vs_analyzer(text):
    p_score = analyzer.polarity_scores(text)
    return p_score['compound']

df['Polarity Score'] = df['Reviews'].apply(vs_analyzer)

# function to analyse
def vader(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound < 0 :
        return 'Negative'
    else:
        return 'Neutral'
df['Result'] = df['Polarity Score'].apply(vader)
df.head()


# In[25]:


vader_counts = df['Result'].value_counts()
vader_counts


# In[ ]:


##Visual Insights


# In[31]:


colors = ['yellow', 'green', 'red']

# Plot the donut chart
fig, ax = plt.subplots()
ax.pie(vader_counts.values, labels=vader_counts.index, explode = (0, 0, 0.25), colors=colors, autopct='%1.2f%%', shadow=True, startangle=140)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add a circle at the center
centre_circle = plt.Circle((0,0), 0.75, color='black', fc='white', linewidth=1)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.show()


# In[30]:


##Wordcloud


# In[27]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(df):
    wordcloud = WordCloud(
        background_color='white',
        stopwords= STOPWORDS,
        max_words=200,
        contour_width=3,
        contour_color='dodgerblue',
        colormap='Dark2',
        max_font_size=30,
        scale=8,
        random_state=1)

    wordcloud=wordcloud.generate(str(df))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.show()

show_wordcloud(df["Reviews"])


# In[29]:


df.to_csv("updatedBA_reviews.csv")


# In[ ]:




