# Spam Detection UI

A colorful and attractive web-based user interface for detecting spam messages using a Naive Bayes classifier trained on SMS data.

## Features

- 🎨 Colorful and modern UI with emojis
- 📝 Input a message and get instant prediction: Spam or Ham (not spam)
- 📊 Displays model accuracy and confusion matrix heatmap
- Uses scikit-learn's MultinomialNB for classification
- Built with Streamlit for easy deployment

## Installation

1. Ensure you have Python installed (3.7+ recommended).
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Open the provided URL in your browser (usually http://localhost:8501).
3. Enter a message in the text area.
4. Click "🔍 Predict" to see if it's spam or not.
5. View the model performance metrics and confusion matrix on the right.

## Dataset

The model is trained on the `spam.csv` dataset, which contains SMS messages labeled as 'ham' or 'spam'.

## Troubleshooting

- If you get encoding errors, ensure `spam.csv` is in the same directory.
- Make sure all dependencies are installed.
- If Streamlit is not recognized, try `python -m streamlit run app.py`.