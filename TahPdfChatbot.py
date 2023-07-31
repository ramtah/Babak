import os
import io
import PyPDF2
#from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

#load_dotenv()

# Adding the title "BABAK" centered at the top
st.markdown("<h1 style='text-align: center;'>BABAK</h1>", unsafe_allow_html=True)

openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
llm = OpenAI(temperature=0, streaming=True, openai_api_key=openai_api_key)
tools = load_tools(["ddg-search"])
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True
)

# Apply custom CSS style for the "Submit" button
st.write("""
<style>
    .stButton>button {
        background-color: green;
        color: white;
        padding: 0.5em 1em;
        border: none;
        border-radius: 0.5em;
    }
</style>
""", unsafe_allow_html=True)

# File uploader for PDF documents
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Check if a PDF file has been uploaded
if uploaded_file is not None:
    # Read the PDF and extract text
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    #st.write("PDF content:")
    #st.write(text)

# Text area for user input
prompt = st.text_area("Enter your question here:")

# Button to submit the question
if st.button("Submit", key="submit_button"):
    if uploaded_file is not None:
        st.write("🧠 thinking...")
        response = agent.run(prompt + " " + text)
        st.write(response)
    else:
        st.warning("Please upload a PDF file first.")
