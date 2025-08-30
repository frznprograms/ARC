import streamlit as st
import requests

st.title("Review Submission Form")

# Create a form
with st.form("review_form"):
    name = st.text_input("Business Name")
    category = st.text_input("Category")
    description = st.text_area("Description")
    review = st.text_area("Review")


    # Option 2: Radio buttons with stars
    rating_radio = st.radio(
        "Rating:",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: "⭐" * x
    )

    submitted = st.form_submit_button("Submit")

if submitted:

    data = {
        "name": name,
        "category": category,
        "description": description,
        "review": review,
        "rating": rating_radio
    }
    response = requests.post("http://127.0.0.1:8000/submit_review/", json=data)
    st.write(response.json())