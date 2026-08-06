import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Data_Preprocessor.preprocessor import DataPreprocessor

MODEL_DIR = ROOT_DIR / "Fake_News_Model_Creator" / "exported_models"
VOCAB_PATH = MODEL_DIR / "vocab.json"
CONFIG_PATH = MODEL_DIR / "model_config.json"
RNN_MODEL_PATH = MODEL_DIR / "rnn_best_model.pt"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_best_model.pt"

class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        final_hidden = hidden[-1]
        return self.fc(self.dropout(final_hidden)).squeeze(-1)

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        if self.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        return self.fc(self.dropout(final_hidden)).squeeze(-1)

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

vocab = load_json(VOCAB_PATH)
config = load_json(CONFIG_PATH)
PAD_IDX = config["pad_idx"]
UNK_IDX = config["unk_idx"]
device = torch.device("cpu")

def load_checkpoint(path):
    return torch.load(path, map_location=device)

def build_rnn():
    checkpoint = load_checkpoint(RNN_MODEL_PATH)
    model = RNNClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def build_lstm():
    checkpoint = load_checkpoint(LSTM_MODEL_PATH)
    model = LSTMClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        bidirectional=config["lstm_bidirectional"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

rnn_model = build_rnn()
lstm_model = build_lstm()

app = FastAPI(title="Fake News Detector")

class NewsInput(BaseModel):
    title: str
    body: str

def preprocess_text(title, body):
    dataframe = pd.DataFrame([{"title": title, "text": body}])
    preprocessor = DataPreprocessor(dataframe, cols=["title", "text"])
    cleaned = preprocessor.run_preprocessor()
    cleaned["combined_text"] = preprocessor.combine_text()
    return cleaned.loc[0, "combined_text"]

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def encode(text):
    token_ids = [vocab.get(token, UNK_IDX) for token in tokenize(text)]
    if not token_ids:
        token_ids = [UNK_IDX]

    return torch.tensor([token_ids], dtype=torch.long), torch.tensor([len(token_ids)])

def predict_with_model(model, text):
    encoded_text, lengths = encode(text)

    with torch.no_grad():
        logit = model(encoded_text.to(device), lengths.to(device))
        real_probability = torch.sigmoid(logit).item()

    label = "REAL" if real_probability >= 0.5 else "FAKE"
    confidence = real_probability if label == "REAL" else 1 - real_probability

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "real_probability": round(real_probability, 4),
        "fake_probability": round(1 - real_probability, 4)
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fake News Detector</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f4f6f8;
      color: #1f2933;
    }
    main {
      max-width: 860px;
      margin: 40px auto;
      padding: 0 20px;
    }
    h1 {
      margin-bottom: 20px;
      font-size: 28px;
    }
    label {
      display: block;
      margin: 16px 0 8px;
      font-weight: 700;
    }
    input, textarea {
      width: 100%;
      box-sizing: border-box;
      padding: 12px;
      border: 1px solid #c7d0d9;
      border-radius: 6px;
      font-size: 15px;
      background: white;
    }
    textarea {
      min-height: 190px;
      resize: vertical;
    }
    button {
      margin-top: 18px;
      padding: 11px 18px;
      border: 0;
      border-radius: 6px;
      background: #1769aa;
      color: white;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }
    #error {
      margin-top: 14px;
      color: #b42318;
      font-weight: 700;
    }
    .results {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 24px;
    }
    .result {
      padding: 16px;
      border: 1px solid #d8dee5;
      border-radius: 8px;
      background: white;
    }
    .model {
      font-size: 13px;
      color: #5b6773;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .label {
      margin-top: 8px;
      font-size: 30px;
      font-weight: 800;
    }
    .fake {
      color: #b42318;
    }
    .real {
      color: #0b7a45;
    }
    .detail {
      margin-top: 10px;
      line-height: 1.6;
      color: #3d4852;
    }
    @media (max-width: 700px) {
      main {
        margin: 22px auto;
      }
      .results {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Fake News Detector</h1>
    <form id="form">
      <label for="title">Title</label>
      <input id="title" name="title" type="text" required>

      <label for="body">News Body</label>
      <textarea id="body" name="body" required></textarea>

      <button id="run" type="submit">Run</button>
    </form>

    <div id="error"></div>
    <section id="results" class="results"></section>
  </main>

  <script>
    const form = document.getElementById("form");
    const button = document.getElementById("run");
    const results = document.getElementById("results");
    const error = document.getElementById("error");

    function renderResult(modelName, result) {
      const labelClass = result.label === "REAL" ? "real" : "fake";
      return `
        <article class="result">
          <div class="model">${modelName}</div>
          <div class="label ${labelClass}">${result.label}</div>
          <div class="detail">
            Confidence: ${(result.confidence * 100).toFixed(2)}%<br>
            Real probability: ${(result.real_probability * 100).toFixed(2)}%<br>
            Fake probability: ${(result.fake_probability * 100).toFixed(2)}%
          </div>
        </article>
      `;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      button.textContent = "Running...";
      error.textContent = "";
      results.innerHTML = "";

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            title: document.getElementById("title").value,
            body: document.getElementById("body").value
          })
        });

        if (!response.ok) {
          throw new Error("Prediction failed");
        }

        const data = await response.json();
        results.innerHTML = renderResult("RNN", data.rnn) + renderResult("LSTM", data.lstm);
      } catch (err) {
        error.textContent = err.message;
      } finally {
        button.disabled = false;
        button.textContent = "Run";
      }
    });
  </script>
</body>
</html>
"""

@app.post("/predict")
def predict(news: NewsInput):
    cleaned_text = preprocess_text(news.title, news.body)

    return {
        "preprocessed_text": cleaned_text,
        "rnn": predict_with_model(rnn_model, cleaned_text),
        "lstm": predict_with_model(lstm_model, cleaned_text)
    }
