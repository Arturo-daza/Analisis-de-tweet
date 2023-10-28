import flask 
from flask import Flask, render_template, request
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem import LancasterStemmer
import os

# Define the file path using os.path.join
model_path = os.path.join('models', 'random_forest.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load('models\preprocesamiento.pkl')

stopword = set(stopwords.words('english'))
stemmer = LancasterStemmer()
def preprocessing(text):
    # print(text)
    try:
        # Convert text to lowercase
        text = text.lower()
        text = re.sub('-',' ',text)   # replace `word-word` as `word word`
        text = re.sub(f'[{string.digits}]',' ',text)  # remove digits
        text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stopword])  # remove stopwords and stem other words
        text =  re.sub(r'@\S+', '',text)                     # remove twitter handles
        text =  re.sub(r'http\S+', '',text)                  # remove urls
        text =  re.sub(r'pic.\S+', '',text)
        text =  re.sub(r"[^a-zA-Z+']", ' ',text)             # only keeps characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text+' ')      # keep words with length>1 only
        text = ''.join([i for i in text if i not in string.punctuation])
        words = nltk.tokenize.word_tokenize(text,language="english", preserve_line=True)
        text = " ".join([i for i in words if i not in stopword and len(i)>2])
        text= re.sub("\s[\s]+", " ",text).strip()            # remove repeated/leading/trailing spaces
    except:
        print('Error:')
        print(text)

    return re.sub(f'[{re.escape(string.punctuation)}]','',text) # remove punctuations
nuevos_datos = []

# Create a Flask app
app = Flask(__name__)

# Define a route and a function to handle it
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    title = "whose tweet is this"

    if request.method == 'POST':
        tweet = request.form['user_input']
        nuevos_datos.append(tweet)
        nuevos_datos[0] = preprocessing(nuevos_datos[0])
        X_nuevos = vectorizer.transform(nuevos_datos)
        nuevos_datos.clear()
        prediction = model.predict(X_nuevos)
        print(prediction)
        if prediction[0] == "Donald J. Trump":
            return render_template('index.html', title=title, tweet= tweet, person = prediction[0])
        else: 
            return render_template('index.html', title=title, tweet= tweet, person = prediction[0])
        
        return f"You entered: {user_input}"

    return render_template('index.html', title=title)

if __name__ == '__main__':
    app.run(debug=True)