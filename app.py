from flask import Flask,  render_template, request
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/submit', methods=['POST', 'GET'])
def analyze_text():
    if request.method == 'POST':
        text = request.form['textArea']
        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans("", "", string.punctuation))
        tokenized_words = word_tokenize(cleaned_text, "english")
        final_words = []
        for word in tokenized_words:
            if word not in stopwords.words('english'):
                final_words.append(word)
        emotion_list = []
        with open('emotion.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", "").replace(",", "").replace("'", "").strip()
                word, emotion = clear_line.split(":")
                if word in final_words:
                    emotion_list.append(emotion)

        w = Counter(emotion_list)
        score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
        neg = score['neg']
        pos = score['pos']
        if neg > pos:
            sentiment = "The text is negative"
        elif neg < pos:
            sentiment = "The text is positive"
        else:
            sentiment = "The text is neutral"
        fig, ax1 = plt.subplots()
        ax1.bar(w.keys(), w.values())
        fig.autofmt_xdate()
        plt.savefig('static/plot.png')
        return render_template('result.html', sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
