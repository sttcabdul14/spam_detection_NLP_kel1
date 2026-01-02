import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import base64

# =========================
# BACKGROUND
# =========================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("bg.jpeg")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("spam_model_id.pkl")
tfidf = joblib.load("tfidf_id.pkl")

# =========================
# UI
# =========================
st.markdown("## 📩 Deteksi Spam SMS Bahasa Indonesia")

pesan = st.text_area("Masukkan isi SMS:")

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi"):
    if pesan.strip() == "":
        st.warning("⚠️ Pesan tidak boleh kosong")
    else:
        vector = tfidf.transform([pesan])
        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0].max()

        if pred.lower() == "spam":
            st.error(f"🚨 SPAM ({prob*100:.2f}%)")
        else:
            st.success(f"✅ HAM ({prob*100:.2f}%)")

        # =========================
        # GRAFIK KATA SPAM
        # =========================
        st.markdown("### 📊 Kata yang Sering Muncul pada Spam")

        df = pd.read_csv("sms_spam_indo.csv")

        spam_text = " ".join(df[df["Kategori"] == "spam"]["Pesan"])
        words = spam_text.lower().split()
        common_words = Counter(words).most_common(10)

        labels, values = zip(*common_words)

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        plt.xticks(rotation=45)
        st.pyplot(fig)
