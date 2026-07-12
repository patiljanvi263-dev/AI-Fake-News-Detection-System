from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import mysql.connector
import os

app = Flask(__name__)

app.secret_key = "fake_news_secret_key"


# ==========================
# MySQL Database Connection
# ==========================

db = mysql.connector.connect(
    host=os.environ.get("DB_HOST", "localhost"),
    user=os.environ.get("DB_USER", "root"),
    password=os.environ.get("DB_PASSWORD", "your_password"),
    database=os.environ.get("DB_NAME", "fake_news_detection"),
    port=int(os.environ.get("DB_PORT", 3306))
)

cursor = db.cursor()


# ==========================
# Load Machine Learning Model
# ==========================

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# ==========================
# Home Page
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# Login
# ==========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        sql = """
        SELECT * FROM users 
        WHERE username=%s AND password=%s
        """

        cursor.execute(sql, (username, password))

        user = cursor.fetchone()

        if user:

            session["username"] = username

            return redirect(url_for("dashboard"))

        else:

            return render_template(
                "login.html",
                error="Invalid Username or Password!"
            )

    return render_template("login.html")


# ==========================
# Register
# ==========================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        fullname = request.form["fullname"]
        email = request.form["email"]
        username = request.form["username"]
        password = request.form["password"]


        sql = """
        INSERT INTO users
        (fullname, email, username, password)
        VALUES (%s,%s,%s,%s)
        """

        values = (
            fullname,
            email,
            username,
            password
        )


        cursor.execute(sql, values)
        db.commit()


        return redirect(url_for("login"))


    return render_template("register.html")



# ==========================
# Dashboard
# ==========================

@app.route("/dashboard")
def dashboard():

    if "username" not in session:
        return redirect(url_for("login"))


    return render_template(
        "dashboard.html",
        username=session["username"]
    )



# ==========================
# Prediction Page
# ==========================

@app.route("/predict")
def predict_page():

    return render_template(
        "predict.html",
        prediction=None,
        confidence=None
    )



# ==========================
# Prediction Result
# ==========================

@app.route("/predict_result", methods=["POST"])
def predict_result():


    if "username" not in session:
        return redirect(url_for("login"))


    username = session["username"]

    news = request.form["news"]


    # Convert text

    news_vector = vectorizer.transform([news])


    # Prediction

    result = model.predict(news_vector)[0]


    probability = model.predict_proba(news_vector)[0]



    if result == 1:

        prediction = "REAL NEWS"
        confidence = round(probability[1] * 100, 2)


    else:

        prediction = "FAKE NEWS"
        confidence = round(probability[0] * 100, 2)



    # Save History

    sql = """
    INSERT INTO prediction_history
    (username, news, result, confidence)
    VALUES (%s,%s,%s,%s)
    """


    values = (
        username,
        news,
        prediction,
        confidence
    )


    cursor.execute(sql, values)

    db.commit()



    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence
    )



# ==========================
# History
# ==========================

@app.route("/history")
def history():


    if "username" not in session:
        return redirect(url_for("login"))


    username = session["username"]


    sql = """
    SELECT news, result, confidence, created_at
    FROM prediction_history
    WHERE username=%s
    ORDER BY created_at DESC
    """


    cursor.execute(sql, (username,))


    records = cursor.fetchall()



    return render_template(
        "history.html",
        records=records
    )



# ==========================
# Profile
# ==========================

@app.route("/profile")
def profile():

    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]

    sql = """
    SELECT fullname, email, username
    FROM users
    WHERE username=%s
    """

    cursor.execute(sql, (username,))

    user = cursor.fetchone()

    return render_template(
        "profile.html",
        user=user
    )

# ==========================
# Logout
# ==========================

@app.route("/logout")
def logout():

    session.pop("username", None)

    return redirect(url_for("login"))



# ==========================
# Run Flask
# ==========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )