from flask import Flask, request, render_template, redirect, url_for
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('results', filename=file.filename))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Read the file content
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    # Compute TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([content])
    feature_names = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_
    tf_values = tfidf_matrix.toarray()[0]

    # Create dataframe
    df = pd.DataFrame({
        'word': feature_names,
        'tf': tf_values,
        'idf': idf_values
    })

    df_sorted = df.sort_values(by='idf', ascending=False).head(50)

    words_data = df_sorted.to_dict(orient='records')

    return render_template('results.html', words=words_data)

if __name__ == '__main__':
    app.run(debug=True)
