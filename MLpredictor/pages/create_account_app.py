#import libraries

import streamlit as st
import sqlite3
import hashlib
import time

# ---------- Database Functions ----------
def create_connection():
    return sqlite3.connect("users.db", check_same_thread=False)

def create_table():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fullname TEXT NOT NULL,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL,
                    password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def add_user(fullname, username, email, password):
    conn = create_connection()
    c = conn.cursor()
    c.execute("INSERT INTO users (fullname, username, email, password) VALUES (?, ?, ?, ?)",
              (fullname, username, email, password))
    conn.commit()
    conn.close()

def user_exists(username):
    conn = create_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result is not None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="Create Account", layout="centered")

# ---------- CSS ----------
st.markdown("""
    <style>
         html, body, [data-testid="stApp"] {
            background-color: rgb(120, 200, 220);
        }
        .main-container {
            background-color: rgb(30, 100, 150);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
            max-width: 500px;
            margin: auto;
        }
        h2 {
            text-align: center;
            color: rgb(50, 130, 184);
        }
        .stButton>button {
            background-color: rgb(50, 130, 184);
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.3s;
            text-align: center;
            margin-left: 40%;
        }
        .stButton>button:hover {
            background-color: rgb(80, 100, 150);
        }
        .create-account>button {
            background-color: #28a745 !important;  /* Green for Create */
            color: white !important;
            margin-right: 30%;
        }
        .create-account>button:hover {
            background-color: rgb(50, 130, 184) !important;
            color: white !important;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.3s;
            text-align: center;
            border: none;
            transform: translateX(-40%);
        }
    </style>
""", unsafe_allow_html=True)

# ---------- App UI ----------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown("##  Create an Account")

create_table()

with st.form("signup_form"):
    fullname = st.text_input("👤 Full Name")
    username = st.text_input("🆔 Username")
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")
    confirm_password = st.text_input("🔁 Confirm Password", type="password")
    with st.markdown('<div class="create-account">', unsafe_allow_html=True):
        submit = st.form_submit_button("Create Account")

    if submit:
        if not fullname or not username or not email or not password or not confirm_password:
            st.warning("⚠️ Please fill in all fields.")
        elif password != confirm_password:
            st.error("❌ Passwords do not match.")
        elif user_exists(username):
            st.error("🚫 Username already exists.")
        else:
            hashed_pw = hash_password(password)
            add_user(fullname, username, email, hashed_pw)
            time.sleep(2)
            st.success("✅ Account created successfully!")
            st.switch_page('pages/login.py')
st.markdown('</div>', unsafe_allow_html=True)
