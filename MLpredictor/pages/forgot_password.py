# pages/forgot_password.py
import streamlit as st
import sqlite3
import random
import smtplib
from email.mime.text import MIMEText

st.set_page_config(page_title="Forgot Password", layout="centered")

# ---------- SECRETS ----------
SENDER = st.secrets["MAIL_SENDER"]
MAIL_PASSWORD = st.secrets["MAIL_PASSWORD"]
MAIL_SERVER = st.secrets["MAIL_SERVER"]
MAIL_PORT = int(st.secrets["MAIL_PORT"])
DB_NAME = st.secrets["DB_NAME"]

# ---------- DB FUNCTIONS ----------
def create_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# ---------- SEND OTP ----------
def send_verification_code(email, code):
    try:
        message = MIMEText(f"Your password reset code is: {code}")
        message["Subject"] = "Password Reset Code"
        message["From"] = SENDER
        message["To"] = email

        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            server.starttls()
            server.login(SENDER, MAIL_PASSWORD)  # App Password must be used here
            server.sendmail(SENDER, email, message.as_string())

        return True
    except Exception as e:
        st.error(f"Error sending OTP: {e}")
        return False

# ---------- UI ----------
st.markdown("<h3 style='text-align:center;'>Forgot Password?</h3>", unsafe_allow_html=True)

email_input = st.text_input("Enter your registered Gmail address:").lower()

if st.button("Send Code"):
    if not email_input:
        st.warning("Please enter your email.")
    else:
        conn = create_connection()
        c = conn.cursor()
        # Look up the correct column: email
        c.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email_input,))
        user = c.fetchone()
        conn.close()

        if user:
            code = str(random.randint(100000, 999999))
            st.session_state["reset_code"] = code
            st.session_state["reset_email"] = email_input

            if send_verification_code(email_input, code):
                st.success("Verification code sent. Check your email.")
                st.switch_page("pages/verify_code.py")
        else:
            st.error("Email not found.")
