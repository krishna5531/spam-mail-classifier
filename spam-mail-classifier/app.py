import streamlit as st
import string
import pickle
import nltk.corpus
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
##stop = stopwords()

##textprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
###modal = pickle.load(open('model.pkl','rd'))
modal = pickle.load(open('model.pkl', 'rb'))



st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the massage")

if st.button('Predict'):
# 1. Preprocess
    transform_sms = transform_text(input_sms)

# 2. Vectorize
    vector_input = tfidf.transform([transform_sms])

# 3. Predict
    result = modal.predict(vector_input)[0]


# 4. Displey
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")