from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import torch
import warnings

warnings.filterwarnings("ignore")

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Om1024/racist-bert")
model = AutoModelForSequenceClassification.from_pretrained("Om1024/racist-bert")

# Mapping of label indices to classes
label_mapping = {0: "Sexist", 1: "Racist", 2: "Others"}

# Streamlit app title
st.title("Racist/Sexist Detection App")
st.write("""
    This app uses a fine-tuned BERT model to classify text as **Racist**, **Sexist**, or **Not**.
    """)

user_input = st.text_area("Enter a comment for classification:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.write("Please enter some text for classification.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(predictions, dim=1).item()

        # Show the prediction
        st.write(f"This Comment is : **{label_mapping[predicted_label]}**")
        st.write(f"Confidence Score: {max(predictions.numpy().tolist()[0])*100}%")