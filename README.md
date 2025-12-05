# AI Email Triage & Autoresponder

An AI-powered system that reads customer emails, classifies them into categories (e.g., billing, complaint, technical issue, sales), and generates suggested replies automatically.  
Built with a FastAPI backend, a Streamlit dashboard, and modern NLP models (BERT + LLM-based response generation).

---

## 🚀 Features

- Email classification using a fine-tuned BERT/DistilBERT model  
- Automatic reply generation using templates + an LLM (e.g., FLAN-T5 / GPT)  
- Confidence scores for predictions to help human agents decide  
- Interactive Streamlit UI for:
  - Single email triage
  - Bulk processing from CSV
- Modular pipeline (preprocessing → classification → reply generation)  
- Designed as a real-world customer support assistant for helpdesk / support teams.

---

## 🧱 Architecture

High-level flow:

1. User submits an email (or multiple emails) via the Streamlit web app  
2. Request is sent to the FastAPI backend  
3. Backend:
   - Cleans and preprocesses the email text  
   - Runs the Email Classifier (BERT) to predict category  
   - Passes the category + email to the Autoresponder Engine
4. Autoresponder:
   - Selects a base template depending on category  
   - Optionally uses an LLM to polish the response  
5. Streamlit displays:
   - Predicted category + confidence  
   - Suggested reply ready to be copied/edited  

---

## 🛠 Tech Stack

Languages: Python  
NLP & ML: HuggingFace Transformers (BERT/DistilBERT), FLAN-T5 / GPT (for replies)  
Backend: FastAPI  
Frontend: Streamlit  
Data: Public email datasets (e.g., Enron) + custom labeled categories  
Other: Pandas, Scikit-learn, Uvicorn

## 📂 Project Structure

Planned folder layout:

```bash
ai-email-triage-autoresponder/
│
├── app/
│   ├── main.py              # FastAPI app (API endpoints)
│   ├── classifier.py        # BERT model loading & inference
│   ├── responder.py         # Reply generation logic (templates + LLM)
│   ├── preprocessing.py     # Email cleaning & preprocessing utilities
│   └── models/
│       └── email_bert_model/  # Fine-tuned classification model
│
├── streamlit_app/
│   └── app.py               # Streamlit UI
│
├── data/
│   ├── raw/                 # Original email dataset(s)
│   └── processed/           # Cleaned / labeled data
│
├── notebooks/
│   └── model_training.ipynb # Training & evaluation experiments
│
├── requirements.txt
└── README.md