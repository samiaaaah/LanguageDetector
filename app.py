import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

modelNB = pickle.load(open('modelNB.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))
cv =CountVectorizer()

st.title("Language Detection App")
def set_background_image(image_url):
    # Apply custom CSS to set the background image
    page_bg_img = '''
    <style>
    .stApp {
        background-position: top;
        background-image: url(%s);
        background-size: cover;
    }

    @media (max-width: 768px) {
        /* Adjust background size for mobile devices */
        .stApp {
            background-position: top;
            background-size: contain;
            background-repeat: no-repeat;
        }
    }
    </style>
    ''' % image_url
    st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    # Set the background image URL
    background_image_url = "https://images.unsplash.com/photo-1634128221889-82ed6efebfc3?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8bGFuZ3VhZ2V8ZW58MHx8MHx8fDA%3D"

    # Set the background image
    set_background_image(background_image_url)

    custom_css = """
       <style>
       body {
           background-color: #4699d4;
           color: #ffffff;
           font-family: Arial, sans-serif;
       }
       h1 {
           color: #ffffff !important; /* Set title color to white */
       }
       select {
           background-color: #000000 !important; /* Black background for select box */
           color: #ffffff !important; /* White text within select box */
       }
       label {
           color: #ffffff !important; /* White color for select box label */
       }
       </style>
       """
    st.markdown(custom_css, unsafe_allow_html=True)


if __name__ == "__main__":

    main()
user_text = st.text_area("Enter a Text:")

if st.button("Detect Language"):
    data = cv.transform([user_text]).toarray()
    output = modelNB.predict(data)
    st.success(f"The detected language is: {output[0]}")
