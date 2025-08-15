import streamlit as st

st.set_page_config(page_title="Redirecting...", layout="centered")

# Redirect to login.py (now inside pages/)
st.switch_page("pages/login.py")
