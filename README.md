# 🧠 AI-Powered Resume Matching Engine

This project is a **Flask-based web application** that matches resumes to job descriptions using **LLMs** and **NLP techniques**. It automates the candidate screening process by ranking resumes based on their similarity to a given job description and generating a brief AI-generated summary of why each candidate is a potential fit.

---

## 🔍 Features

- ✅ **Job-Resume Matching** using cosine similarity on vector embeddings
- 🤖 **LLM-Powered Summaries** via AWS Bedrock's Claude 3 to justify candidate fit
- 📊 **Sorted Matching Results** with similarity scores and summaries
- 🌐 **Web UI** built using Flask, Bootstrap, and Jinja templates
- 🔐 Safe handling of AWS credentials (excluded in public repo)

---

## 🛠️ Tech Stack

- **Frontend**: HTML 
- **Backend**: Python, Flask  
- **AI Integration**: Claude 3 via **AWS Bedrock**
- **Vector Similarity**: Sentence Transformers + Cosine Similarity  
- **NLP**: Preprocessing using NLTK / spaCy (optional)

---
