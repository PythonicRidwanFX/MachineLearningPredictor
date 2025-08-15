# pages/reset_password.py
import streamlit as st
import sqlite3
import hashlib

st.set_page_config(page_title="Reset Password", layout="centered")

# Ensure OTP was verified before allowing reset
if not st.session_state.get("otp_verified"):
    st.error("You must verify your OTP before resetting your password.")
    st.stop()

st.markdown("<h3 style='text-align:center;'>Create New Password</h3>", unsafe_allow_html=True)

new_password = st.text_input("New Password", type="password")
confirm_password = st.text_input("Confirm Password", type="password")

def create_connection():
    return sqlite3.connect("users.db", check_same_thread=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

if st.button("Update Password"):
    if new_password != confirm_password:
        st.error("Passwords do not match.")
    elif len(new_password) < 6:
        st.error("Password must be at least 6 characters long.")
    else:
        conn = create_connection()
        c = conn.cursor()
        hashed_pw = hash_password(new_password)
        c.execute("UPDATE users SET password=? WHERE email=?", (hashed_pw, st.session_state["reset_email"]))
        conn.commit()
        conn.close()

        st.success("Password updated successfully! You can now log in.")

# Instead of clearing otp_verified here, let login page handle it
st.session_state["allow_login_after_reset"] = True
st.switch_page("pages/login.py")
