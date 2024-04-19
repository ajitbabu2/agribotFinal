import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from googletrans import Translator
from streamlit_mic_recorder import speech_to_text
import speech_recognition as sr

import json
from PIL import Image

import numpy as np
import tensorflow as tf

# Initialize the recognizer
recognizer = sr.Recognizer()
translator = Translator()

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(question, language):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # new_db = FAISS.load_local(
    #     "faiss_index", embeddings, allow_dangerous_deserialization=True
    # )
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )

    # print(response)
    if language == "English":
        st.write("Reply: ", response["output_text"])
    elif language == "Tamil":
        translator = Translator()
        translated_text = translator.translate(
            response["output_text"], src="en", dest="ta"
        )
        st.write("Reply (Tamil): ", translated_text.text)


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype("float32") / 255.0
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


def main():

    st.set_page_config("Agribot")
    st.header("Agri Helpline Bot 🧑🏻‍🌾")
    tab1, tab2 = st.tabs(["Chatbot", "Classifier"])
    with tab1:
        st.subheader("Chatbot")
        # Add content for App 1 here
        user_question = st.text_input("Ask a question to your Agribot agent")
        text = speech_to_text(
            language="en-US",  # Make sure to use a supported language code
            start_prompt="English",  # Button text to start recording
            stop_prompt="Stop recording",  # Button text to stop recording
            just_once=True,  # Change to True if you want to limit it to one recording per session
            use_container_width=False,
            key="eng",
        )
        tamil_text = speech_to_text(
            language="ta-IN",  # Make sure to use a supported language code
            start_prompt="தமிழ்",  # Button text to start recording
            stop_prompt="Stop recording",  # Button text to stop recording
            just_once=True,  # Change to True if you want to limit it to one recording per session
            use_container_width=False,
            key="tam",
        )

        if tamil_text:
            translated_text = translator.translate(tamil_text, dest="en")
            user_input(translated_text.text, "Tamil")
            user_question = ""
        if text:
            user_input(text, "English")
            user_question = ""

        if user_question:
            user_input(user_question, "English")
            user_question = ""

        # with st.sidebar:
        #     st.title("Menu:")
        #     pdf_docs = st.file_uploader(
        #         "Upload your PDF Files and Click on the Submit & Process Button",
        #         accept_multiple_files=True,
        #     )
        #     if st.button("Submit & Process"):
        #         with st.spinner("Processing..."):
        #             raw_text = get_pdf_text(pdf_docs)
        #             text_chunks = get_text_chunks(raw_text)
        #             get_vector_store(text_chunks)
        #             st.success("Done")

    with tab2:
        st.subheader("🌿 Plant Disease Classifier")
        # st.title('🌿 Plant Disease Classifier')

        uploaded_image = st.file_uploader(
            "Upload an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((150, 150))
                st.image(resized_img)

            with col2:
                if st.button("Classify"):
                    # Preprocess the uploaded image and predict the class
                    prediction = predict_image_class(
                        model, uploaded_image, class_indices
                    )
                    st.success(f"Prediction: {str(prediction)}")


if __name__ == "__main__":
    main()
