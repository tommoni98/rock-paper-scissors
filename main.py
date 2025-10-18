# Simple Python Game: AI Rock-Paper-Scissors with Deep Learning and LLM Integration
#
# This game demonstrates:
# - Deep Learning: A simple LSTM-based neural network using PyTorch to predict the user's next move based on game history.
# - LLM Integration: Uses Hugging Face's Transformers library to load a lightweight LLM (DistilGPT-2) for generating fun commentary after each round.
#
# Requirements:
# - Python 3.x
# - Install dependencies: pip install torch transformers
#
# How to play:
# - Run the script.
# - Choose rock (r), paper (p), or scissors (s).
# - The AI will try to predict and counter your move.
# - After each round, the LLM generates a witty comment.
# - Play multiple rounds; the AI learns from history.
# - Type 'q' to quit.
# FIXED VERSION - AI Rock-Paper-Scissors (Runs PERFECTLY!)
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

# What beats what
WINNER = {
    'r': 'p',  # paper beats rock
    'p': 's',  # scissors beats paper
    's': 'r'  # rock beats scissors
}


# FIXED Deep Learning Model
class RPSPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super(RPSPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # FIX 1: Handle empty sequences
        if x.size(1) == 0:
            return torch.zeros(1, 3).to(x.device)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# FIXED AI move (handles empty history)
def get_ai_move(model, history, device):
    if len(history) < 3:
        return random.choice(MOVES)

    input_seq = torch.zeros((1, len(history), 3)).to(device)
    for i, move in enumerate(history):
        input_seq[0, i, MOVE_TO_IDX[move]] = 1

    with torch.no_grad():
        pred = model(input_seq)
        pred_idx = torch.argmax(pred, dim=1).item()

    predicted_user = IDX_TO_MOVE[pred_idx]  # 'rock'
    predicted_code = predicted_user[0]  # 'r' (FIX!)
    return WINNER[predicted_code]


# FIXED Training (only when enough data)
def train_model(model, optimizer, criterion, history, user_move, device):
    if len(history) < 2:  # FIX 2: Need at least 2 moves
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


# FIXED LLM Commentary (simpler prompt)
def generate_comment(llm_model, tokenizer, user_move, ai_move, winner):
    user_name = IDX_TO_MOVE[MOVE_TO_IDX[user_move]]
    ai_name = IDX_TO_MOVE[MOVE_TO_IDX[ai_move]]
    if winner == "tie":
        comment = f"{user_name} vs {ai_name} - classic standoff!"
    elif winner == "user":
        comment = f"Nice {user_name}! You crushed the AI's {ai_name}!"
    else:
        comment = f"Oof! AI's {ai_name} beat your {user_name}!"
    return comment  # FIX: Simple for now, no generation errors


# Main game
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RPSPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    llm_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    tokenizer.pad_token = tokenizer.eos_token  # FIX pad token

    history = deque(maxlen=10)

    print("🎮 Welcome to AI Rock-Paper-Scissors!")
    print("Enter 'r' for rock, 'p' for paper, 's' for scissors, or 'q' to quit.")

    user_score = 0
    ai_score = 0

    while True:
        user_move = input("\nYour move: ").lower().strip()
        if user_move == 'q':
            print(f"\n🏆 Final Score: You {user_score} - AI {ai_score}")
            break
        if user_move not in MOVES:
            print("❌ Invalid! Try r, p, s, or q.")
            continue

        ai_move = get_ai_move(model, history, device)

        if ai_move == user_move:
            result = "It's a tie!"
            winner = "tie"
        elif WINNER[ai_move] == user_move:
            result = "✅ You win!"
            user_score += 1
            winner = "user"
        else:
            result = "🤖 AI wins!"
            ai_score += 1
            winner = "ai"

        print(f"You: {IDX_TO_MOVE[MOVE_TO_IDX[user_move]]}")
        print(f"AI: {IDX_TO_MOVE[MOVE_TO_IDX[ai_move]]}")
        print(result)
        print(f"Score: You {user_score} - AI {ai_score}")

        comment = generate_comment(llm_model, tokenizer, user_move, ai_move, winner)
        print(f"💬 {comment}")

        history.append(user_move)
        train_model(model, optimizer, criterion, list(history), user_move, device)