import streamlit as st
import sqlite3
import hashlib

st.set_page_config(page_title="Login Page", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            background-color: rgb(180, 200, 220);
        }
        .stButton>button {
            background-color: #007BFF !important;
            color: white !important;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0056b3 !important;
        }
        .marquee-text {
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
            font-weight: bold;
            box-sizing: border-box;
            animation: marquee 20s linear infinite;
            font-size: 20px;
            color: black;
        }
        @keyframes marquee {
            0%   { text-indent: 100% }
            100% { text-indent: -100% }
        }
        .forgot-password {
            text-align: center;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- DB FUNCTIONS ----------
def create_connection():
    return sqlite3.connect("users.db", check_same_thread=False)

def get_user_by_username(username):
    conn = create_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result

def verify_password(input_password, stored_hashed_password):
    return hashlib.sha256(input_password.encode()).hexdigest() == stored_hashed_password

# ---------- PAGE CONTENT ----------
st.markdown("""
   <div class="marquee-text">
        This Website is working for predicting dataset model (Machine Learning)
   </div>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; color:black;'>Login</h2>", unsafe_allow_html=True)

# ---------- LOGIN FORM ----------
with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        login = st.form_submit_button("Login")

if login:
    if username and password:
        user_data = get_user_by_username(username)
        if user_data:
            stored_hashed_password = user_data[4]  # adjust if your DB column differs
            if verify_password(password, stored_hashed_password):
                st.success("Login successful")
                st.switch_page("pages/index.py")
            else:
                st.error("Incorrect password")
        else:
            st.error("Username not found")
    else:
        st.error("Please enter both username and password")

# ---------- FORGOT PASSWORD BUTTON ----------
st.markdown('<div class="forgot-password">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    if st.button("Forgot Password?"):
        st.switch_page("pages/forgot_password.py")

# ---------- CREATE ACCOUNT BUTTON ----------
col1, col2, col3 = st.columns([2, 1, 1])
with col3:
    create = st.button("Create Account")

if create:
    st.switch_page("pages/create_account_app.py")


if st.session_state.get("allow_login_after_reset"):
    st.session_state["otp_verified"] = False
    del st.session_state["allow_login_after_reset"]
