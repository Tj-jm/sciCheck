import numpy as np
import pandas as pd
import streamlit as st
import joblib
import time
import matplotlib.pyplot as plt

# App title
st.set_page_config(page_title="SciCheck", layout="centered")


# Load model once
@st.cache_resource
def load_model(name):
    if name == "Naive Bayes":
        return joblib.load('./model/sci_model.pkl')
    elif name == "Logistic Regression":
        return joblib.load('./model/logistic_model.pkl')
    else:
        raise ValueError("Unsupported model selected")
model_choice = st.selectbox("Select a Classification Model", ["Naive Bayes", "Logistic Regression"])
model_pipeline = load_model(model_choice)
st.caption(f"Model used: `{model_choice}`")
# model_pipeline = load_model()


st.title("SciCheck: Abstract Classifier")
st.write("Enter a research abstract below to predict its scientific category.")

# Initialize state variables
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "prediction" not in st.session_state:
    st.session_state.prediction = ""
if "proba" not in st.session_state:
    st.session_state.proba = []
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Input
input_box = st.empty()
user_input = st.text_area("Paste Your Abstract Here",
                                 value=st.session_state.text_input,
                               
                           height=300)

# Predict button
if st.button("Predict Category"):
    if user_input.strip() == "":
        st.error("Please enter a valid abstract.")
    else:
        prediction = model_pipeline.predict([user_input])[0]
        proba = model_pipeline.predict_proba([user_input])[0]

        # Store in session state
        st.session_state.predicted = True
        st.session_state.prediction = prediction
        st.session_state.proba = proba

# Clear button — outside prediction block
if st.button("Clear"):
    st.session_state.predicted = False
    st.session_state.prediction = ""
    st.session_state.proba = []
    st.session_state.text_input = ""  
    
       

# Show prediction if already done
if st.session_state.predicted:
    st.success(f"**Predicted Category**: `{st.session_state.prediction}`")
    show_probs = st.checkbox("Show all category probabilities")
    if show_probs:
        labels = model_pipeline.classes_
        st.dataframe({
            "Category": labels,
            "Probability (%)": [round(p * 100, 2) for p in st.session_state.proba]
        })
    show_graph=st.checkbox("compare probability of categories")
    if show_graph:
          labels = model_pipeline.classes_
          probabilities = [round(p * 100, 2) for p in st.session_state.proba]

          
          df_probs = pd.DataFrame({
               "Category": model_pipeline.classes_,
               "Probability": [round(p * 100, 2) for p in st.session_state.proba],
               "Confidence": ["High" if p > 0.5 else "Low" for p in st.session_state.proba]
               })
      
          st.bar_chart(df_probs, x="Category", y="Probability", color="Confidence")

  
st.markdown("---")
st.header("📂 Bulk Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload a CSV with 'abstract' and 'category' columns", type="csv")

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)

    if 'abstract' not in df_test.columns or 'category' not in df_test.columns:
        st.error("CSV must contain 'abstract' and 'category' columns.")
    else:
        # Predict
        y_true = df_test['category']
        y_pred = model_pipeline.predict(df_test['abstract'])
        y_proba = model_pipeline.predict_proba(df_test['abstract'])

        # Show predictions
        df_test['Predicted'] = y_pred
        st.subheader("Prediction Results")
        st.dataframe(df_test[['abstract', 'category', 'Predicted']])

        # Classification Report
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

        report = classification_report(y_true, y_pred, output_dict=True)
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix Plot
        st.subheader("Confusion Matrix")
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred, labels=model_pipeline.classes_)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model_pipeline.classes_,
                    yticklabels=model_pipeline.classes_,
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
