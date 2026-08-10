from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd

app = Flask(__name__)
app.secret_key = "mysecretkey123"     # keep it safe

model = joblib.load("model.joblib")
prediction_history = []

# ---------------------- DATABASE HELPERS ------------------------
def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# ---------------------- AUTH ROUTES -----------------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_pw = generate_password_hash(password)

        conn = get_db()
        try:
            conn.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                         (username, email, hashed_pw))
            conn.commit()
        except:
            return "User already exists or error occurred"

        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))
        else:
            return "Invalid email or password"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------- MAIN PAGES -----------------------------

@app.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("predict.html", username=session["username"])

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            data = {
                "MedInc": float(request.form["MedInc"]),
                "HouseAge": float(request.form["HouseAge"]),
                "AveRooms": float(request.form["AveRooms"]),
                "AveBedrms": float(request.form["AveBedrms"]),
                "Population": float(request.form["Population"]),
                "AveOccup": float(request.form["AveOccup"]),
                "Latitude": float(request.form["Latitude"]),
                "Longitude": float(request.form["Longitude"])
            }

            input_df = pd.DataFrame([data])

            pred = model.predict(input_df)[0]

            # Convert MedHouseVal to actual price
            actual_price = pred * 100000  
            formatted_price = f"{actual_price:,.2f}"

            return render_template("predict.html", prediction=formatted_price)

        except Exception as e:
            return f"Error: {e}"

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
