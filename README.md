# 🎮 AI Rock-Paper-Scissors Game
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

An interactive Rock-Paper-Scissors game where you play against an AI that learns your patterns using Deep Learning!

## 🚀 Play Online
👉 **[Play Now on Streamlit](https://your-app-url.streamlit.app)**

---
## 🧠 How It Works
- **Deep Learning:** LSTM neural network predicts your next move
- **Pattern Recognition:** AI learns from your playing history
- **Real-time Training:** Model updates after each round

---
## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** PyTorch (LSTM)
- **Language:** Python 3.8+

---
## 📦 Local Installation
```bash
# Clone the repository
git clone https://github.com/tommoni98/rock-paper-scissors.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---
## 🎯 Features
- ✅ Real-time score tracking
- ✅ AI that learns your patterns
- ✅ Fun commentary after each round
- ✅ Clean, responsive UI
- ✅ Pattern history visualization

---
## 📊 ML Model Details
- **Architecture:** LSTM (64 hidden units)
- **Input:** One-hot encoded move history (last 10 moves)
- **Output:** Predicted next move (rock/paper/scissors)
- **Training:** Real-time gradient descent after each round

---
## 📄 **License**

This project is open source and available under the [MIT License](LICENSE).

---