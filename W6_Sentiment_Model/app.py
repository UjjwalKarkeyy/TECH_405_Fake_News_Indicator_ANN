import re
import json
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

def clean(t):
  # replace https with " "
  t = re.sub(r"http\S+l@\w+", " ", t)
  t = re.sub(r"[^a-zA-Z\s]", " ", t)
  return t.lower().split()

def encode(t, maxlen=30):
  ids = [vocab.get(w, 1) for w in clean(t)][:maxlen]
  return ids + [0]*(maxlen - len(ids))

class SentimentRNN(nn.Module):
  def __init__(self, vocab_size, emb=64, hidden=128):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, emb)
    self.rnn = nn.RNN(emb, hidden, batch_first=True)
    self.fc = nn.Linear(hidden, 2)

  def forward(self, x):
    x = self.embedding(x)
    x, _ = self.rnn(x)
    return self.fc(x[:, -1]) # last hidden memory

try:
  with open('vocab.json', 'r') as f:
    vocab = json.load(f)
except Exception as e:
  print(f'Exception occurred: {e}')

model = SentimentRNN(len(vocab))
model.load_state_dict(torch.load('rnn.pt', map_location=torch.device('cpu')))

app = FastAPI()

class Tweet(BaseModel):
  text: str

@app.post("/predict")
def predict(tweet: Tweet):
    x = torch.tensor([encode(tweet.text)])
    with torch.no_grad():
        y = model(x)
        pred = torch.argmax(y, dim=1).item()
    return {'sentiment': 'positive' if pred==1 else 'negative'}


