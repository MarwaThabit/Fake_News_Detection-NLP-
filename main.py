import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import pickle
from sklearn.preprocessing import LabelEncoder
from model import *


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    tokens_pos = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    tokens_lemmatized = []
    for word, pos in tokens_pos:
        if pos.startswith('V'):  
            lemma = lemmatizer.lemmatize(word, pos='v')
        elif pos.startswith('N'):  
            lemma = lemmatizer.lemmatize(word, pos='n')
        elif pos.startswith('J'):  
            lemma = lemmatizer.lemmatize(word, pos='a')
        else:
            lemma = lemmatizer.lemmatize(word)
        tokens_lemmatized.append(lemma)
    text = ' '.join(tokens_lemmatized)
    return text

def text_embedding(x_train,x_test):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    return tfidf_train,tfidf_test
            
def main():
    
    df = pd.read_csv('news.csv',low_memory=False,usecols=['title', 'text', 'label'])
    df = df.dropna(subset=['text', 'label'])
    
    # DuplicateNumber=df.duplicated().sum()
    # print(DuplicateNumber)
    #  df.drop_duplicates(inplace=True)
    
    df['content'] = df['title'] + ' ' + df['text']
    df['content'] = df['content'].apply(preprocess_text)
    
    labels = df['label']
    lbl = LabelEncoder()
    lbl.fit(labels)
    labels = lbl.transform(labels)
    
    df['label']=labels
    df.to_csv('output.csv',index=False)
    
    x_train, x_test, y_train, y_test = train_test_split(df['content'], labels, test_size=0.2, random_state=0,shuffle=False)
    
    tfidf_train,tfidf_test=text_embedding(x_train,x_test)
    
    train_accuracy = passive_model(tfidf_train, y_train)
    print("="*55)
    print(f'                Training_Accuracy: {round(train_accuracy * 100, 2)}%')

    
    with open('tfidf_test.pkl', 'wb') as f:
      pickle.dump(tfidf_test, f)

    with open('y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)
    

main()

