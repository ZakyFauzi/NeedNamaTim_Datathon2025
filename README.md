# Datathon 2025: Sentiment Analysis by NeedNamaTim

## Deskripsi
Repositori ini berisi kode pengembangan untuk proyek analisis sentimen ulasan pelanggan menggunakan model Support Vector Machine (SVM) dan Naive Bayes. Proyek ini menggunakan PRDECT-ID Dataset (5.400 ulasan) untuk melatih model yang mengklasifikasikan produk ke dalam kategori Bagus, Normal, atau Buruk, serta menghasilkan insight actionable seperti rekomendasi perbaikan pengiriman atau kualitas produk.

## Struktur Repositori
- `Notebook_NeedNamaTim.ipynb`: Notebook Jupyter dengan kode lengkap untuk preprocessing, pelatihan model, dan pembuatan insight (eksperimen prototype sederhana).
- `requirements.txt`: Daftar dependensi Python dengan versi spesifik.
- Model dan file preprocessing (`svm_model.pkl`, `nb_model.pkl`, `le_sentiment.pkl`, `le_emotion.pkl`, `scaler.pkl`) tersedia di [Hugging Face](https://huggingface.co/username/sentiment-analysis-tokopedia).
- Dataset tersedia di [Hugging Face](https://huggingface.co/username/PRDECT-ID).

## Instalasi
1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/sentiment-analysis-tokopedia.git
   cd sentiment-analysis-tokopedia


Buat environment virtual (opsional):python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Instal dependensi:pip install -r requirements.txt


Download dataset dari Hugging Face atau Mendeley Data.

Penggunaan

Buka newnotebook.ipynb di Jupyter Notebook atau JupyterLab.
Pastikan file dataset (PRDECT-ID.csv) ada di direktori yang sama.
Jalankan sel-sel kode untuk:
Memproses dataset (ekstraksi fitur dengan VADER dan text2emotion).
Melatih model SVM dan Naive Bayes.
Menghasilkan insight berbasis aturan.
Membuat visualisasi (word cloud, confusion matrix).



Contoh Kode Prediksi:
import pandas as pd
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import text2emotion as te
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model dan preprocessing
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_sentiment = pickle.load(open('le_sentiment.pkl', 'rb'))
le_emotion = pickle.load(open('le_emotion.pkl', 'rb'))

# Fungsi ekstraksi fitur
sid = SentimentIntensityAnalyzer()
def extract_sentiment(review):
    if pd.isna(review):
        return 'Positive'
    scores = sid.polarity_scores(str(review))
    return 'Positive' if scores['compound'] >= 0 else 'Negative'

def extract_emotion(review):
    if pd.isna(review):
        return 'Happy'
    try:
        emotions = te.get_emotion(str(review))
        dominant_emotion = max(emotions, key=emotions.get) if emotions else 'Happy'
        return dominant_emotion if dominant_emotion in ['Happy', 'Love', 'Anger', 'Fear', 'Sadness'] else 'Happy'
    except:
        return 'Happy'

# Contoh data baru
data_baru = pd.DataFrame({
    'Customer Review': ['Produk sangat bagus, baterai tahan lama, pengiriman cepat'],
    'Customer Rating': [5]
})
data_baru['Sentiment'] = data_baru['Customer Review'].apply(extract_sentiment)
data_baru['Emotion'] = data_baru['Customer Review'].apply(extract_emotion)
data_baru['Sentiment_Encoded'] = le_sentiment.transform(data_baru['Sentiment'])
data_baru['Emotion_Encoded'] = le_emotion.transform(data_baru['Emotion'])

# Prediksi
features = ['Customer Rating', 'Sentiment_Encoded', 'Emotion_Encoded']
data_scaled = scaler.transform(data_baru[features])
prediksi = svm_model.predict(data_scaled)
print(prediksi)  # Output: ['Bagus']

Dependensi
Lihat requirements.txt untuk daftar lengkap dependensi. Versi utama:

pandas==2.2.2
nltk==3.8.1
text2emotion==0.0.5
scikit-learn==1.5.1
matplotlib==3.9.1
seaborn==0.13.2
wordcloud==1.9.3
