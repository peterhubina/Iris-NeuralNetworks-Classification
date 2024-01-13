#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


# Load Iris Dataset
# X - features (data)
# y - triedy (setosa, versicolor, virginica)
# Structured dataset (rows, columns)
iris = load_iris()
X = iris.data
y = iris.target

epochs=100


# In[3]:


print("Dataset shape:", X.shape)
print("Labels:", set(y))
print("Classes:", iris['target_names'])

iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
iris_df.head()


# In[4]:


# Rozdelenie datasetu na trenovaciu a testovaciu cast (test_size = 0.3 znamena 30% testovacia cast)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[5]:


# Vizualizacia datasetu pomocou bodoveho grafu
def create_scatter_plot(X, y, features, title):
    # Set up the figure and axis
    plt.figure(figsize=(12, 10))
    plt.suptitle(title, fontsize=16)

    # Vizualizacia kazdeho paru features
    plot_number = 1
    for i in range(len(features)):
        for j in range(len(features)):
            if i < j:
                plt.subplot(2, 3, plot_number)
                for target in np.unique(y):
                    subset = X[y == target]
                    plt.scatter(subset[:, i], subset[:, j], label=iris.target_names[target])
                plt.xlabel(features[i])
                plt.ylabel(features[j])
                plt.legend()
                plot_number += 1

    plt.show()


# In[6]:


# Vizualizacia trenovacieho setu
create_scatter_plot(X_train, y_train, iris.feature_names, 'Iris Dataset - Training Set')


# In[7]:


# Vizualizacia testovacieho setu
create_scatter_plot(X_test, y_test, iris.feature_names, 'Iris Dataset - Testing Set')


# In[8]:


# Standardizacia datasetu
scaler = StandardScaler()
# fit_transform - vypocitanie mean a standard deviation, nasleduje transformacia trenovacic dat
X_train = scaler.fit_transform(X_train)
# tranformacia testovacich dat pomocou rovnakej mean a sd value
X_test = scaler.transform(X_test)


# In[9]:


# Konvertovanie na tensory
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# In[10]:


# Vytvorenie dataset objektov
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Vytvorenie loaderov pre loadovanie dat pocas treningu, poskytovanie dat v kazdej iteracii podla batch_size
# shuffle = True pri trenovani pre zredukovanie rizika overfittingu
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# In[11]:


# MLP Model: Strukturovane datasety
# CNN Model: Spracovanie obrazkov..., spatial data
# RNN Model: NLP, voice recognition..., sekvencne data


# In[12]:


# Definovanie MLP Modelu
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        # 4 input features, 8 neuronov
        self.fc1 = nn.Linear(4, 10)
        # Hidden layer - 10 neuronov
        self.fc2 = nn.Linear(10, 10)
        # output layer, 3 output features
        self.fc3 = nn.Linear(10, 3)

    # Metoda pre predikciu, zavola sa po vlozeni dat do modelu
    def forward(self, x):
        x = self.fc1(x)
        # Aplikovanie relu pre hidder layer
        # rectified linear unit
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[13]:


# Definovanie CNN Modelu
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Vytvorenie konvolucnej vrstvy, feature extraction
        # atributy - 1 input channel, 16 output channels, filter size 3 (filter nad datami)
        self.conv1 = nn.Conv1d(1, 16, 3)
        # 1D pooling layer - window size 2, vypocet maximalnej hodnoty z dvoch skumanych prvkov na vstupe,
        # redukovanie data size na polovicu, konsoliduje data
        self.pool = nn.MaxPool1d(2)
        # 1 feature size, 3 output size (3 triedy)
        self.fc1 = nn.Linear(16, 3)

    def forward(self, x):
        # Add a channel dimension
        x = x.unsqueeze(1)
        # Aplikovanie relu na konvolucnu vrstvu
        x = torch.relu(self.conv1(x))
        # pooling layer
        x = self.pool(x)
        # Transformacia tenzoru z pooling layer na 1D
        x = x.view(x.size(0), -1)
        # Aplikovanie Linearnej vrstvy na vytvorenie outputu
        x = self.fc1(x)
        return x


# In[14]:


# Definovanie RNN Modelu
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        # long-short term memory layer
        # single feature
        # 8 hidden units (8 neuronov)
        # batch_first specifikuje input shape (batch_size, sequence_length, input_size)
        self.rnn = nn.LSTM(input_size=1, hidden_size=8, batch_first=True)
        # Linearna vrstva, output 3 classy
        self.fc = nn.Linear(8, 3)

    def forward(self, x): 
        # Prida input_size do tensora (predtym batch_size, sequence_length)
        x = x.unsqueeze(2)
        # Pass data through LSTM layer
        x, _ = self.rnn(x)
        # Take the last time step
        x = x[:, -1, :]
        # Output z LSTM layer do Linear Layer (fully connected, shape (batch_size, 3))
        x = self.fc(x)
        return x


# In[15]:


# Trenovaci loop, pouziva back propagation
def train(model, criterion, optimizer, epoch_number):
    losses_mean = []
    loss_vall_arr = []
    acc_arr = []

    final_epochs = 0
    prev_accuracy = 0.0
    same_accuracy_count = 0

    precisions = []
    recalls = []
    
    for epoch in range(epoch_number):
        losses = 0
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            # Ziska output pre dany input iteracie
            outputs = model(inputs)
            # vypocet loss medzi predikciami a skutocnymi labelmi
            loss = criterion(outputs, labels)
            # Pripocita loss
            losses += loss.detach().numpy()
            # Vypocet gradientov
            loss.backward()
            # Update modelu na zaklade vypocitanych gradientov
            optimizer.step()
        # Priemerny loss pre kazdu epochu trenovania
        avg_loss = losses / len(train_loader)
        
        losses_mean.append(avg_loss)

        # Evaluacia pre kazdu epochu, vracia accuracy a loss value
        acc, loss_vall, precision, recall = evaluate_model(model, train_loader)
        acc_arr.append(acc)
        loss_vall_arr.append(loss_vall)
        final_epochs += 1

        print(f'Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {acc}%, Precision: {precision}, Recall: {recall}')

        precisions.append(precision)
        recalls.append(recall)

        # Check for consecutive loss increases
        if epoch > 0 and avg_loss > losses_mean[-2]:
            increase_count += 1
            if increase_count >= 3:
                print("Stopping early due to increasing loss.")
                break
        else:
            increase_count = 0  # Reset the counter if loss decreases
        
    return losses_mean, loss_vall_arr, acc_arr, precisions, recalls, final_epochs


# In[16]:


# Funkcia pre vyhodnotenie modelu na testovacich datach
def evaluate_model(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    losses = 0
    losses_mean = []

    predicted_labels = []
    true_labels = []
    
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses += loss.detach().numpy()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_labels.extend(predicted.numpy())
            true_labels.extend(labels.numpy())

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted')
            
    losses_mean.append(losses/len(train_loader))
    accuracy = 100 * correct / total
    return accuracy, losses_mean, precision, recall


# In[17]:


def visualise(loss_vall_arr, acc_arr, precisions, recalls, epochs):
    epochs = range(0, epochs)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(epochs, loss_vall_arr, label='valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs,acc_arr, label='acc', color = 'orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs, precisions, label='precision', color = 'red')
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs, recalls, label='recall', color = 'blue')
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[18]:


mlp_model = MLPModel()
cnn_model = CNNModel()
rnn_model = RNNModel()

# Definovanie loss funkcie a optimalizatorov

# sluzi na vypocet loss 
criterion = nn.CrossEntropyLoss()

# Optimalizacny algoritmus Adam
# learning rate 0.001
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

# Trenovanie modelov
print("MLP")
losses_mean, loss_vall_arr, acc_arr, precisions, recalls, final_epochs  = train(mlp_model, criterion, mlp_optimizer, epochs)
visualise(loss_vall_arr, acc_arr, precisions, recalls, final_epochs)
print("CNN")
losses_mean, loss_vall_arr, acc_arr, precisions, recalls, final_epochs  = train(cnn_model, criterion, cnn_optimizer, epochs)
visualise(loss_vall_arr, acc_arr, precisions, recalls, final_epochs)
print("RNN")
losses_mean, loss_vall_arr, acc_arr, precisions, recalls, final_epochs  = train(rnn_model, criterion, rnn_optimizer, epochs)
visualise(loss_vall_arr, acc_arr, precisions, recalls, final_epochs)


# In[19]:


def classify_samples(model, features_tensor, labels_tensor):
    # Nastavi model na evaluation mode
    model.eval()
    # requires_grad false
    with torch.no_grad():
        # Predikcia
        predictions = model(features_tensor)
        # Ziska predikovanu class
        _, predicted_classes = torch.max(predictions, 1)

    # Porovnanie predikcii so skutocnymi datami
    for i, (pred, actual) in enumerate(zip(predicted_classes, labels_tensor)):
        predicted_label = iris['target_names'][pred]
        actual_label = iris['target_names'][actual]
        print(f"Sample {i+1}: Predicted - {predicted_label}, Actual - {actual_label}")


# In[20]:


# Testuje nové vzorky na MLP modeli, transformuje ich, konvertuje na tenzory, predpovedá triedy a vypisuje názvy predpovedaných tried

new_samples = [
    ([5.1, 3.5, 1.4, 0.2], 0),
    ([6.7, 3.0, 5.2, 2.3], 2),
    ([5.9, 3.0, 4.2, 1.5], 1),
    ([6.1, 2.9, 4.7, 1.4], 1),
    ([4.6, 3.4, 1.4, 0.3], 0)
]

# Rozdelenie features a labels, tuple unpacking
new_features, new_labels = zip(*new_samples)

# Rovnaka scaling transformation ako pre trenovacie data
features_scaled = scaler.transform(np.array(new_features))

# Convert to tensors
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
labels_tensor = torch.tensor(new_labels, dtype=torch.long)

print("MLP")
classify_samples(mlp_model, features_tensor, labels_tensor)
print("CNN")
classify_samples(cnn_model, features_tensor, labels_tensor)
print("RNN")
classify_samples(rnn_model, features_tensor, labels_tensor)

