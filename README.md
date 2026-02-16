# 🛡️ JobShield AI  
### Detecting Fake Job Offers to Prevent Human Trafficking (Recruitment-Stage Prevention)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-green)
![License](https://img.shields.io/badge/License-Academic%20Use-yellow)

JobShield AI is an AI-powered system that detects **fake job postings** and **suspicious recruitment messages** to help prevent **human trafficking and exploitation** at the recruitment stage.

It uses **Machine Learning + NLP + Threat Scoring + Trafficking Signal Detection** and provides explainable results through an interactive **Streamlit dashboard**.

---

## 🏆 Achievement

 **First Prize Winner**  
 **SVKM’s NMIMS, Chandigarh**  
 **“Empowering Women and Girls in Science” Workshop + Ideation Competition**  
 **11 February 2026 (Wednesday)**  
 **Project: JobShield AI**

---

## ✨ Key Features

###  Fake Job Classification (ML Model)
- Classifies job text as **Fake Job / Real Job**
- Uses **TF-IDF + Logistic Regression**

###  Risk Level + Confidence Score
- Shows risk level: **Low / Medium / High / Critical**
- Based on prediction probability

###  Threat Score (0–100)
A hybrid scoring system combining:
- ML probability score  
- Rule-based scam indicators  

###  Trafficking Signal Detection
Detects patterns like:
- “Urgent hiring”
- “No interview”
- “Passport required”
- “Free visa”
- “Agent will manage everything”
- “Girls required abroad”
- “Pay registration / training fee”

###  Contact + Location Extraction
Extracts:
- Phone numbers  
- Emails  
- URLs  
- Location mentions (Dubai, Qatar, etc.)

###  Explainable AI Output
Shows suspicious keywords (**ML risk indicators**) responsible for prediction.

###  PDF Job Ad Scanner
- Upload PDF job advertisement
- Extracts text → analyzes it

###  Batch Scan (CSV Upload)
- Upload CSV containing job messages
- Scans all records automatically

###  Database + History (SQLite)
- Stores every scan with timestamp
- History tab to view old scans

###  Alerts System
- High-risk cases generate persistent alerts
- Mark alerts as **Resolved / False Positive**

###  Evidence Report Generator (TXT + PDF)
Downloads a full evidence report including:
- prediction  
- risk level  
- threat score  
- extracted contacts  
- trafficking signals  
- safety guidance  

---

##  Machine Learning Concepts Used

- Text preprocessing (cleaning + normalization)
- NLP feature extraction using **TF-IDF**
- Supervised learning (**Binary Classification**)
- Logistic Regression classifier
- Train-test split
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Confidence scoring using prediction probability
- Explainability using top TF-IDF keywords
- Hybrid detection system: **ML + Rule-based threat scoring**

---

##  How It Works (Pipeline)

1. User inputs job offer text / message / PDF
2. Text preprocessing (lowercase, remove URLs, remove punctuation)
3. Convert to TF-IDF vector
4. ML model predicts Fake/Real + confidence
5. Rule engine detects scam patterns
6. Final threat score is generated (0–100)
7. Trafficking signals + extracted contacts + locations shown
8. Report is generated and stored in database

---

##  Tech Stack

**Frontend/UI**
- Streamlit

**Machine Learning**
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

**Database**
- SQLite

**Data**
- Pandas
- NumPy

**PDF Support**
- pdfplumber (text extraction)
- reportlab (PDF report generation)

---

##  Project Structure

```bash
JobShield-AI/
│── app.py
│── model_utils.py
│── requirements.txt
│── fake_job_postings.csv
│── JobShieldAI_project.ipynb
│── jobshield.db
│── .gitignore
│── README.md

```
---
## Installation & Setup
```bash
1️ Clone Repository
git clone https://github.com/your-username/jobshield-ai.git
cd jobshield-ai

2️ Create Virtual Environment
python -m venv venv

3️ Activate Virtual Environment

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

4️ Install Requirements
pip install -r requirements.txt

5️ Run Streamlit App
streamlit run app.py
```
---
## Dataset Used

- This project is trained using a Fake Job Posting Dataset containing labeled job descriptions (Fake/Real).
You can replace it with your own dataset for improved accuracy.
---

## Example Input
- Urgent hiring! High salary 80,000/month.
 Location will be shared after registration fee.
  Submit Aadhaar copy for confirmation.
---

## Future Scope 

-Deploy as a web service for NGOs and recruitment platforms

-Integrate WhatsApp/SMS scam monitoring using APIs

-Add multilingual support (Hindi, Punjabi, Marathi, etc.)

-Build real-time scam heatmap for smart cities

-Improve explainability using SHAP / LIME

-Add blacklist database of scam phone numbers + URLs

-Create mobile app version for rural users

---

## Disclaimer

- This tool is built for educational and safety awareness purposes.
It does not replace legal verification or official investigation.
Always verify job offers using trusted sources.
---

## Author

-Sehaj Kaur
-Project: JobShield AI

