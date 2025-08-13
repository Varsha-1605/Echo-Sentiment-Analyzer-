# 🎤 Alexa Reviews Sentiment Analysis

An interactive **Streamlit** application that uses **Machine Learning** to classify Amazon Alexa reviews as **Positive** or **Negative**.  
Trained on real Alexa feedback data, it provides instant predictions, confidence levels, and a probability breakdown — all in a clean, user-friendly interface.

---

## 📌 Features
- **Real-time Sentiment Analysis** — analyze as you type
- **Probability & Confidence Scores** — see how confident the model is
- **Interactive UI** with sample reviews for quick testing
- **Detailed Preprocessing Insights** — view tokenization, stopword removal, and lemmatization
- **Beautiful Visuals** — probability distribution charts
- **Reusable ML Pipeline** — includes preprocessing, model training, and prediction scripts

---

## 🛠️ Tech Stack
- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **ML Model**: Random Forest Classifier
- **Vectorization**: CountVectorizer (Bag-of-Words)
- **Libraries**: Pandas, scikit-learn, NLTK, Altair

---

## 📂 Project Structure
```

├── app.py                  # Streamlit app (UI + prediction logic)
├── model.py                # Model training & saving
├── text_preprocessing.py   # Custom text preprocessing pipeline
├── models/                 # Trained model & vectorizer files (.pkl)
├── data/                   # Dataset files
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

````

---

## 📈 Model Performance
| Metric          | Score   |
|-----------------|---------|
| Train Accuracy  | 99.39%  |
| Test Accuracy   | 94.19%  |
| Precision       | 94.46%  |
| Recall          | 99.51%  |
| F1 Score        | 96.92%  |

The model achieves **high recall** and **balanced precision**, making it reliable for detecting both positive and negative Alexa reviews.

## 🚀 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/alexa-reviews-sentiment-analysis.git
cd alexa-reviews-sentiment-analysis
````

### 2️⃣ Create & activate virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download NLTK resources (only once)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 5️⃣ Train the model (optional)

If you want to retrain:

```bash
python model.py
```

### 6️⃣ Run the app

```bash
streamlit run app.py
```

---


## 📄 Dataset

This model is trained on the **Amazon Alexa Reviews dataset**, which contains verified customer reviews of Alexa devices.


---

## 💡 Author

**Varsha Dewangan**
🚀 Passionate about AI, Machine Learning, and Web Development
