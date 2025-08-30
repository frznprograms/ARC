import streamlit as st
import requests

st.set_page_config(page_title="Review Analyzer", page_icon="📝", layout="centered")

st.title("📝 Review Analyzer")

# Input fields
name = st.text_input("Business/Location Name")
category = st.text_input("Category")
description = st.text_area("Description")
review_text = st.text_area("Review")
rating = st.slider("Rating", 1, 5, 3)

if st.button("Analyze Review"):
    if not (name and category and description and review_text and rating):
        st.warning("⚠️ Please fill in all fields before analyzing.")
    else:
        try:
            # Construct review JSON
            review_json = {
                "name": name,
                "category": category,
                "description": description,
                "review": review_text,
                "rating": rating,
            }

            # Send to backend
            response = requests.post(
                "http://127.0.0.1:8000/analyze_review/",
                json={"review": review_json},
            )

            if response.status_code == 200:
                logs = response.json().get("logs", [])

                st.subheader("Pipeline Logs")
                if logs:
                    for log in logs:
                        if log["type"].lower() == "success":
                            st.success(f"✅ {log['message']}")
                        elif log["type"].lower() == "warning":
                            st.warning(f"⚠️ {log['message']}")
                        else:
                            st.text(log["message"])
                else:
                    st.info("ℹ️ No logs returned.")
            else:
                st.error(f"❌ Backend error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Error connecting to backend: {e}")
