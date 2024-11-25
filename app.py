import streamlit as st
import pandas as pd
from nltk.tag import CRFTagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import urllib.request  # Untuk mengunduh file

# Unduh model CRF jika belum ada
MODEL_URL = "https://raw.githubusercontent.com/dhavinaocxa/latihan-datmin/main/all_indo_man_tag_corpus_model.crf.tagger"
MODEL_PATH = "all_indo_man_tag_corpus_model.crf.tagger"

try:
    # Cek apakah file model sudah ada
    with open(MODEL_PATH, "r") as f:
        pass
except FileNotFoundError:
    st.write("Mengunduh model CRF...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Judul aplikasi
st.title("POS Tagging, Filter Nouns, dan Analisis Sentimen")

# Upload file
uploaded_file = st.file_uploader("Upload dataset CSV Anda", type=["csv"])
if uploaded_file is not None:
    try:
        # Baca file CSV
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # Sesuaikan encoding dengan dataset Anda
        st.write("Dataset asli:")
        st.dataframe(data.head())  # Tampilkan dataset asli

        # Asumsi kolom tweet bernama 'tweet'
        if 'tweet' in data.columns:
            tweets = data['tweet']

            # POS Tagging
            st.write("Proses POS Tagging...")
            ct = CRFTagger()
            ct.set_model_file(MODEL_PATH)  # Path ke model CRF

            # Tagging setiap tweet
            tagged_tweets = [ct.tag_sents([tweet.split()])[0] for tweet in tweets]

            # Tambahkan hasil POS tagging ke dataset
            data['pos_tagging'] = tagged_tweets

            # Filter hanya nouns
            filtered_tweets = []
            for tagged in tagged_tweets:
                nouns = [word for word, tag in tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
                filtered_tweets.append(" ".join(nouns))

            data['filtered_nouns'] = filtered_tweets

            # Periksa dan hapus nilai kosong
            st.write("Memeriksa nilai kosong pada dataset...")
            data.dropna(subset=['filtered_nouns'], inplace=True)

            if data.empty:
                st.error("Dataset tidak memiliki data yang valid setelah memfilter nilai kosong!")
            else:
                # Klasifikasi Sentimen
                st.write("Proses Analisis Sentimen...")
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(data['filtered_nouns'])  # Menggunakan hasil filter noun

                # Data dummy untuk pelatihan
                dummy_data = {
                    "text": ["bagus", "buruk", "sangat menyenangkan", "sangat mengecewakan", "oke saja"],
                    "sentiment": ["positif", "negatif", "positif", "negatif", "netral"]
                }
                dummy_df = pd.DataFrame(dummy_data)

                # Proses dummy ke fitur dan label
                dummy_X = vectorizer.fit_transform(dummy_df['text'])
                dummy_y = dummy_df['sentiment']

                # Train model
                model = MultinomialNB()
                model.fit(dummy_X, dummy_y)

                # Prediksi sentimen
                data['predicted_sentiment'] = model.predict(X)  # Tambahkan hasil prediksi ke dataset

                # Tampilkan dataset hasil tagging dan sentimen
                st.write("Dataset dengan hasil POS tagging dan prediksi sentimen:")
                st.dataframe(data[['tweet', 'pos_tagging', 'filtered_nouns', 'predicted_sentiment']])

                # Download dataset hasil
                csv = data.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="Download Dataset Hasil",
                    data=csv,
                    file_name="sentiment_analysis_tweets.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Dataset Anda harus memiliki kolom 'tweet'.")
    except FileNotFoundError:
        st.error("Model CRF tidak ditemukan. Pastikan file model tersedia.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
