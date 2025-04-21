import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from pypdf import PdfReader
import os

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "xxxx"

# ============================
# STEP 1: PDF TEXT EXTRACTION
# ============================

@st.cache_data
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ============================
# STEP 2: TEXT CHUNKING
# ============================

@st.cache_resource
def chunk_text(raw_text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([raw_text])

# ============================
# STEP 3: VECTOR EMBEDDING
# ============================

@st.cache_resource
def create_vectorstore(docs):  # FIXED: added docs as parameter
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local("medical_vectorstore")
    return vectordb

# ============================
# STEP 4: LOAD VECTORSTORE + QA
# ============================

@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local("medical_vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

# ============================
# STEP 5: TRAIN SYMPTOM CHECKER
# ============================
@st.cache_resource
def train_symptom_model():
    df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")

    # Reduce data to avoid MemoryError
    df = df.sample(n=30000, random_state=42)

    y = df["diseases"]
    X = df.drop("diseases", axis=1)

    # Use lighter Random Forest settings
    model = RandomForestClassifier(n_estimators=50, n_jobs=1, max_depth=20)
    model.fit(X, y)

    return model, X.columns.tolist()

# ============================
# STEP 6: PREDICT FUNCTION
# ============================

def predict_disease(symptom_text, model, symptom_columns):
    # Convert user input into feature vector
    input_vector = [1 if any(symptom in col for symptom in symptom_text.lower().split()) else 0 for col in symptom_columns]
    prediction = model.predict([input_vector])[0]
    return prediction


# ============================
# STREAMLIT UI STARTS HERE
# ============================

st.set_page_config(page_title="Medical Q&A & Symptom Checker", layout="centered")
st.title("Medical Q&A Chatbot + Symptom Checker")

# Upload PDF once
if not os.path.exists("medical_vectorstore"):
    with st.spinner("Processing Medical PDF..."):
        pdf_text = extract_text_from_pdf("C:\\Users\\Divya\\OneDrive\\Desktop\\dhivya\\medic_bot\\The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf")
        docs = chunk_text(pdf_text)
        create_vectorstore(docs)

qa_chain = load_qa_chain()
model, vectorizer = train_symptom_model()

# Q&A Section
st.header(" Ask a Medical Question")
user_query = st.text_input("Enter your question (e.g. 'What are the symptoms of asthma?')")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Searching medical documents..."):
            result = qa_chain.run(user_query)
            st.success("Answer:")
            st.write(result)

# Symptom Checker Section
st.header("Symptom Checker")
symptom_input = st.text_input("Enter your symptoms (e.g. 'fever and body pain')")

if st.button("Predict Disease"):
    if symptom_input:
        with st.spinner("Predicting disease..."):
            disease = predict_disease(symptom_input, model, vectorizer)
            st.success("Possible Disease:")
            st.write(disease)
