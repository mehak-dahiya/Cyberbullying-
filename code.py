

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import pennylane as qml
import nltk
nltk.download('punkt')


df = pd.read_csv("/Users/mac/Documents/project/final_dataset_hinglish_with_Clean_Text.csv")

possible_labels = ["cyberbullying_type", "label", "labels", "target", "class", "Label"]
label_column = next((col for col in possible_labels if col in df.columns), None)
possible_texts = ["text", "clean_text", "Clean_Text", "content", "tweet", "message", "Text"]
text_column = next((col for col in possible_texts if col in df.columns), None)

X = df[text_column].astype(str).tolist()
y, label_mapping = pd.factorize(df[label_column])
num_classes = len(label_mapping)

print(" Using label:", label_column)
print(" Using text:", text_column)
print(" Classes:", list(label_mapping))


plt.figure(figsize=(6,4))
sns.countplot(x=y, hue=y, palette="Set2", legend=False)
plt.xticks(ticks=range(len(label_mapping)), labels=label_mapping)
plt.title("Class Count")
plt.savefig("class_count.png")
plt.show()


text_all = " ".join(X)
wc = WordCloud(width=800, height=400, background_color="white").generate(text_all)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.savefig("wordcloud.png")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = bert(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :])
    return torch.cat(all_embeddings, dim=0)

X_train_embed = get_bert_embeddings(X_train)
X_test_embed = get_bert_embeddings(X_test)

X_train_np = X_train_embed.numpy()
X_test_np = X_test_embed.numpy()


results = {}
def show_results(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"\n🔹 {model_name} Accuracy: {acc:.2f}")
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in label_mapping]))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_mapping, yticklabels=label_mapping)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.show()  

    results[model_name] = {"Accuracy": acc, "F1": f1}


lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_np, y_train)
show_results("Logistic Regression", y_test, lr.predict(X_test_np))

rf = RandomForestClassifier()
rf.fit(X_train_np, y_train)
show_results("Random Forest", y_test, rf.predict(X_test_np))

svc = SVC()
svc.fit(X_train_np, y_train)
show_results("SVC", y_test, svc.predict(X_test_np))

svm = LinearSVC()
svm.fit(X_train_np, y_train)
show_results("Linear SVM", y_test, svm.predict(X_test_np))

nb = GaussianNB()
nb.fit(X_train_np, y_train)
show_results("Naive Bayes", y_test, nb.predict(X_test_np))


class LSTMClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_layers, n_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)  
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i % len(inputs)], wires=i)
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLSTMClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_layers, n_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.q_params = nn.Parameter(0.01 * torch.randn(1, n_qubits))
        self.fc = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        rnn_out = lstm_out[:, -1, :]
        q_outs = []
        for i in range(rnn_out.size(0)):
            q_input = rnn_out[i][:n_qubits]
            q_out = quantum_circuit(q_input, self.q_params)
            q_outs.append(torch.tensor(q_out, dtype=torch.float32))
        q_outs = torch.stack(q_outs)
        return self.fc(q_outs)


def train_and_evaluate(model, train_loader, test_loader, name, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{name} | Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")


    print("\n=== Evaluating", name, "===")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            y_pred.extend(torch.argmax(preds, dim=1).tolist())
            y_true.extend(yb.tolist())

    show_results(name, y_true, y_pred)


embed_dim = X_train_embed.shape[1]

train_dataset = [(X_train_embed[i], y_train[i]) for i in range(len(y_train))]
test_dataset = [(X_test_embed[i], y_test[i]) for i in range(len(y_test))]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


lstm_model = LSTMClassifier(embed_dim, hidden_dim=128, n_layers=2, n_classes=num_classes)
train_and_evaluate(lstm_model, train_loader, test_loader, "LSTM Classifier", epochs=5)

qlstm_model = QuantumLSTMClassifier(embed_dim, hidden_dim=64, n_layers=1, n_classes=num_classes)
train_and_evaluate(qlstm_model, train_loader, test_loader, "Quantum-LSTM Classifier", epochs=3)


models = list(results.keys())
accuracy_scores = [results[m]["Accuracy"] for m in models]
f1_scores = [results[m]["F1"] for m in models]

x = range(len(models))
plt.figure(figsize=(10,6))
plt.bar(x, accuracy_scores, width=0.4, label="Accuracy", align="center")
plt.bar([i+0.4 for i in x], f1_scores, width=0.4, label="F1-score", align="center")
plt.xticks([i+0.2 for i in x], models, rotation=45)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("model_performance_comparison.png")
plt.show()
