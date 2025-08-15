# pages/verify_code.py
import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText

st.set_page_config(page_title="Verify OTP", layout="centered")

# ---------- FUNCTION TO SEND CODE ----------
def send_verification_code(email, code):
    sender = "youremail@example.com"
    password = "yourpassword"

    msg = MIMEText(f"Your OTP code is: {code}")
    msg["Subject"] = "Password Reset Verification Code"
    msg["From"] = sender
    msg["To"] = email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, email, msg.as_string())

# ---------- PAGE TITLE ----------
st.markdown("<h3 style='text-align:center;'>Enter OTP Code</h3>", unsafe_allow_html=True)

# ---------- OTP INPUT ----------
otp_input = st.text_input("Enter the OTP sent to your email:")

# ---------- VERIFY BUTTON ----------
if st.button("Verify"):
    if "reset_code" not in st.session_state or "reset_email" not in st.session_state:
        st.error("No OTP request found. Please go back and request a code.")
    else:
        if otp_input == st.session_state["reset_code"]:
            st.success("OTP verified successfully!")
            st.session_state["otp_verified"] = True
            st.switch_page("pages/reset_password.py")  # ✅ Correct path
        else:
            st.error("Incorrect OTP. Try again.")

# ---------- RESEND OTP BUTTON ----------
if st.button("Resend OTP"):
    if "reset_email" in st.session_state:
        new_code = str(random.randint(100000, 999999))
        st.session_state["reset_code"] = new_code
        send_verification_code(st.session_state["reset_email"], new_code)
        st.success("New OTP sent to your email.")
    else:
        st.error("No email found. Please request a code first.")
