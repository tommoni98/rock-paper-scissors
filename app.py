# app.py - AI Rock-Paper-Scissors with Streamlit GUI
import streamlit as st
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define moves
MOVES = ['r', 'p', 's']
MOVE_TO_IDX = {'r': 0, 'p': 1, 's': 2}
IDX_TO_MOVE = {0: 'rock', 1: 'paper', 2: 'scissors'}
MOVE_EMOJI = {'r': '🪨', 'p': '📄', 's': '✂️'}

# What beats what
WINNER = {
    'r': 'p',
    'p': 's',
    's': 'r'
}


# Deep Learning Model
class RPSPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super(RPSPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.size(1) == 0:
            return torch.zeros(1, 3).to(x.device)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.session_state.model = RPSPredictor().to(st.session_state.device)
    st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=0.01)
    st.session_state.criterion = nn.CrossEntropyLoss()
    st.session_state.history = deque(maxlen=10)
    st.session_state.user_score = 0
    st.session_state.ai_score = 0
    st.session_state.round_number = 0
    st.session_state.last_result = None
    st.session_state.initialized = True


def get_ai_move(model, history, device):
    if len(history) < 3:
        return random.choice(MOVES)

    input_seq = torch.zeros((1, len(history), 3)).to(device)
    for i, move in enumerate(history):
        input_seq[0, i, MOVE_TO_IDX[move]] = 1

    with torch.no_grad():
        pred = model(input_seq)
        pred_idx = torch.argmax(pred, dim=1).item()

    predicted_user = IDX_TO_MOVE[pred_idx]
    predicted_code = predicted_user[0]
    return WINNER[predicted_code]


def train_model(model, optimizer, criterion, history, user_move, device):
    if len(history) < 2:
        return

    input_seq = torch.zeros((1, len(history) - 1, 3)).to(device)
    for i, move in enumerate(history[:-1]):
        input_seq[0, i, MOVE_TO_IDX[move]] = 1

    target = torch.tensor([MOVE_TO_IDX[history[-1]]]).to(device)

    output = model(input_seq)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def generate_comment(user_move, ai_move, winner):
    user_name = IDX_TO_MOVE[MOVE_TO_IDX[user_move]]
    ai_name = IDX_TO_MOVE[MOVE_TO_IDX[ai_move]]
    if winner == "tie":
        comments = [
            f"{user_name} vs {ai_name} - classic standoff!",
            "Great minds think alike! 🤝",
            "Perfectly matched! Try again!"
        ]
    elif winner == "user":
        comments = [
            f"Nice {user_name}! You crushed the AI's {ai_name}! 🎉",
            "You're on fire! 🔥",
            f"The AI didn't see that {user_name} coming!"
        ]
    else:
        comments = [
            f"Oof! AI's {ai_name} beat your {user_name}! 🤖",
            "The AI is learning your patterns! 🧠",
            "Better luck next round!"
        ]
    return random.choice(comments)


def play_round(user_move):
    st.session_state.round_number += 1

    # Get AI move
    ai_move = get_ai_move(
        st.session_state.model,
        list(st.session_state.history),
        st.session_state.device
    )

    # Determine winner
    if ai_move == user_move:
        result = "tie"
        result_text = "🤝 It's a Tie!"
        result_color = "blue"
    elif WINNER[ai_move] == user_move:
        result = "user"
        result_text = "🎉 You Win!"
        result_color = "green"
        st.session_state.user_score += 1
    else:
        result = "ai"
        result_text = "🤖 AI Wins!"
        result_color = "red"
        st.session_state.ai_score += 1

    # Generate comment
    comment = generate_comment(user_move, ai_move, result)

    # Store result
    st.session_state.last_result = {
        'user_move': user_move,
        'ai_move': ai_move,
        'result': result,
        'result_text': result_text,
        'result_color': result_color,
        'comment': comment
    }

    # Update history and train
    st.session_state.history.append(user_move)
    train_model(
        st.session_state.model,
        st.session_state.optimizer,
        st.session_state.criterion,
        list(st.session_state.history),
        user_move,
        st.session_state.device
    )


# Streamlit UI
st.set_page_config(
    page_title="AI Rock-Paper-Scissors",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 AI Rock-Paper-Scissors")
st.markdown("**Play against an AI that learns your patterns using Deep Learning!**")

# Score display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Your Score", st.session_state.user_score, delta=None)
with col2:
    st.metric("Round", st.session_state.round_number, delta=None)
with col3:
    st.metric("AI Score", st.session_state.ai_score, delta=None)

st.divider()

# Game buttons
st.subheader("Choose Your Move:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🪨 Rock", key="rock", use_container_width=True, type="primary"):
        play_round('r')
        st.rerun()

with col2:
    if st.button("📄 Paper", key="paper", use_container_width=True, type="primary"):
        play_round('p')
        st.rerun()

with col3:
    if st.button("✂️ Scissors", key="scissors", use_container_width=True, type="primary"):
        play_round('s')
        st.rerun()

# Display last result
if st.session_state.last_result:
    st.divider()
    result = st.session_state.last_result

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### You: {MOVE_EMOJI[result['user_move']]} {IDX_TO_MOVE[MOVE_TO_IDX[result['user_move']]]}")
    with col2:
        st.markdown(f"### AI: {MOVE_EMOJI[result['ai_move']]} {IDX_TO_MOVE[MOVE_TO_IDX[result['ai_move']]]}")

    if result['result_color'] == 'green':
        st.success(result['result_text'])
    elif result['result_color'] == 'red':
        st.error(result['result_text'])
    else:
        st.info(result['result_text'])

    st.markdown(f"💬 *{result['comment']}*")

# Game info
st.divider()
with st.expander("ℹ️ How It Works"):
    st.markdown("""
    **Deep Learning AI:**
    - Uses an LSTM neural network to learn your playing patterns
    - Predicts your next move based on game history
    - Gets smarter the more you play!

    **Features:**
    - Real-time score tracking
    - AI commentary after each round
    - Pattern recognition improves over time

    **Tech Stack:**
    - PyTorch for deep learning
    - Streamlit for the UI
    - Transformers for language generation
    """)

# Reset button
if st.button("🔄 Reset Game", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Footer
st.divider()
st.markdown("Made with ❤️ using Streamlit & PyTorch")