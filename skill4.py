import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import re
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

df=pd.read_csv('toxic_train.csv')[['comment_text','toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
def clean(text):
    text=text.lower()
    text=re.sub(r"[^\w\s]"," ",text)
    return text
df['cleaned_text']=df['comment_text'].apply(clean)

vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(df['cleaned_text']).toarray()
y=df[['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
class ToxicDataset(Dataset):
    def __init__(self, features, labels):
        self.X=torch.tensor(features,dtype=torch.float32)
        self.y=torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader=DataLoader(ToxicDataset(X_train,y_train),batch_size=32,shuffle=True)
test_loader=DataLoader(ToxicDataset(X_test,y_test),batch_size=64)

class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(5000, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model(x)

model=NNModel()
loss_fn=nn.BCELoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)
n_epochs=5
train_loss=[]

model.train()
for epoch in range(n_epochs):
    total_loss=0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred=model(X_batch)
        loss=loss_fn(y_pred, y_batch);
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss=total_loss/len(train_loader)
    train_loss.append(avg_loss)
    print(f"Epoch {epoch+1}/{n_epochs} - Loss:{avg_loss:.4f}")

y_predictions=[]
y_true=[]

model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred=model(X_batch)
        y_predictions.append(y_pred.numpy())
        y_true.append(y_batch.numpy())

y_predictions=np.vstack(y_predictions)
y_true=np.vstack(y_true)
y_pred_binary=(y_predictions>0.5).astype(int)
acc=accuracy_score(y_true, y_pred_binary)
roc_aucs=[roc_auc_score(y_true[:,i], y_predictions[:,i]) for i in range(y.shape[1])]

print(f"Accuracy: {acc:.4f}")
for i, label in enumerate(['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    print(f"ROC-AUC: ({label}):{roc_aucs[i]:.4f}")

plt.plot(train_loss,label='Training Loss',marker='o')
plt.title('Epochs Vs Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()