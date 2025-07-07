from flask import Flask, render_template, request
from FilmRecommender import recommend  # kendi fonksiyonun

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    title = None

    if request.method == "POST":
        title = request.form.get("film")
        results = recommend(title)

    return render_template("index.html", title=title, results=results)

if __name__ == "__main__":
    app.run(debug=True)
