import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        color: #ff6b6b;
        font-size: 3em;
        text-align: center;
    }
    .subtitle {
        color: #4ecdc4;
        font-size: 1.5em;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="title">🚀 Spam Detection App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect spam messages with AI! 🤖</p>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Message")
    message = st.text_area("Type your message here:", height=100)

    if st.button("🔍 Predict", key="predict"):
        if message.strip():
            msg_vec = vectorizer.transform([message])
            prediction = model.predict(msg_vec)[0]
            probabilities = model.predict_proba(msg_vec)[0]
            prob_ham = probabilities[0]
            prob_spam = probabilities[1]
            
            if prediction == 1:
                st.error("🚨 This message is SPAM!")
            else:
                st.success("✅ This message is HAM (not spam).")
            
            st.write(f"**Ham Probability:** {prob_ham:.2%}")
            st.write(f"**Spam Probability:** {prob_spam:.2%}")
        else:
            st.warning("⚠️ Please enter a message.")

with col2:
    st.subheader("📊 Model Performance")
    st.metric("Accuracy", f"{accuracy:.2%}")
    
    # Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)