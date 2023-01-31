---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: env
    language: python
    name: python3
---

# ADME Prediction

---

In this notebook we will load in the CYP P450 2C19 Inhibition dataset from Therapeutic Data Commons, and use three different models to predict CYP2C19 inhibition given a drug's SMILE string:

1. A graph neural network
2. A random forest
3. A transformer

We compute the accuracy of each of these models and compare them at the end of the notebook.

### Imports and Dependencies

---

```python
import torch
import torch.nn.functional as F

from tdc.chem_utils import MolConvert
from tdc.single_pred import ADME

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import scatter

from sklearn.ensemble import RandomForestClassifier

from transformers import (
    AdamW,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

from src.logging import log_results
```

### Load Dataset

---

Before training any models, we load the CYP P450 2C19 Inhibition dataset from Therapeutic Data Commons.

```python
# Load CYP P450 2C19 
data = ADME(name="CYP2C19_Veith")

# Separate train and test set
split = data.get_split()
train_set, test_set = split["train"], split["test"]
```

### Graph Neural Network

---

The first model we consider is a graph neural network. In order to do this, we need to convert the drugs' SMILE strings into graph format (specifically PyG graphs as we will use PyTorch-Geometric).

```python
# SMILE string - PyG graph converter
converter = MolConvert(src="SMILES", dst="PyG")

# Convert SMILE strings to graphs
X_train = converter(train_set["Drug"].to_list())
X_test = converter(test_set["Drug"].to_list())

# Cast labels as PyTorch tensors
y_train = torch.tensor(train_set["Y"], dtype=torch.float32)
y_test = torch.tensor(test_set["Y"], dtype=torch.float32)
```

We put our training data into DataLoaders so we can train with minibatches.

```python
# Create data loaders by zipping features and labels together
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=256)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=1)
```

Next, we define a simple graph convolutional network (GCN) class.

```python
class GCN(torch.nn.Module):
    """ Graph convolutional network class. """

    def __init__(self, num_node_features):
        """ Class constructor. Define GCN and linear layers. """
        super().__init__()
        self.num_node_features = num_node_features
        self.conv1 = GCNConv(self.num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, data):
        """ Forward pass. """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc(x).squeeze()

        return F.sigmoid(x)
```

Now we can train our model using the batched training data. This might take around a minute to run.

```python
# Instantate the model & set to training mode
model = GCN(num_node_features=X_train[0].x.shape[1])
model.train()

# Use ADAM optimiser
optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

# 100-epoch training loop with binary cross-entropy loss
for epoch in range(100):
    for batch in train_loader:
        inputs, labels = batch
        optimiser.zero_grad()
        predictions = model(inputs)
        loss = F.binary_cross_entropy(predictions, labels)
        loss.backward()
        optimiser.step()
```

And compute the predictive accuracy on the test set.

```python
# Set model back to evaluation mode
model.eval()

# Count the number of correctly predicted labels on the test set
correct = 0
for datapoint in test_loader:
    input, label = datapoint
    pred = model(input)
    correct += round(float(pred)) == int(label)

# Compute and print the test set accuracy
gcn_acc = int(correct) / int(len(y_test))
print(f"GCN accuracy: {gcn_acc:.4f}")
```

### Random Forest

---

The next model we consider is a random forest. The graph representations of the drugs we used for the graph neural network aren't as useful for tree-based methods; instead, we convert the SMILE strings to MACCS format.

```python
# SMILE string - MACCS converter
converter = MolConvert(src="SMILES", dst="MACCS")

# Convert SMILE strings to MACCS
X_train = converter(train_set["Drug"].to_list())
X_test = converter(test_set["Drug"].to_list())

# Cast labels as numpy arrays
y_train = train_set["Y"].to_numpy()
y_test = test_set["Y"].to_numpy()
```

Now we can train our random forest model.

```python
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
```

Anc compute its accuracy on the test set.

```python
# Compute & print predictive accuracy
preds = clf.predict(X_test)
rf_acc = sum(preds == y_test) / len(y_test)
print(f"Random forest accuracy: {rf_acc:.4f}")
```

### Transformer

---

The final model we consider is a transformer model. We use DistilBERT (a smaller, faster version of BERT) from Hugging Face to do sequence classification using contextualised embeddings of the SMILE strings. The first step is to tokenise & embed our SMILE strings.

```python
# DistilBERT tokeniser
tokeniser = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenise/embed the SMILE strings
train_encodings = tokeniser(train_set["Drug"].to_list(), truncation=True, padding=True)
test_encodings = tokeniser(test_set["Drug"].to_list(), truncation=True, padding=True)

# Cast labels as numpy arrays
train_labels = train_set["Y"].to_numpy()
test_labels = test_set["Y"].to_numpy()
```

Next, we create a custom dataset class to use with PyTorch dataloaders.

```python
class SMILESDataset(torch.utils.data.Dataset):
    """ Custom dataset class. """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
```

And use our custom dataset class to create a training and test dataloader for minibatching.

```python
# Create custom training and test set instances
train_dataset = SMILESDataset(train_encodings, train_labels)
test_dataset = SMILESDataset(test_encodings, test_labels)

# Create dataloaders for training and evaluation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
```

We can now fine-tune DistilBERT on our dataset. To do this properly, we would probably need to let it train for quite a bit longer than 1 epoch, but a single epoch already takes ~30-40 mins.

```python
# Instantate the model & set to training mode
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.train()

# Use ADAM optimiser
optimizer = AdamW(model.parameters(), lr=1e-4)

# Single epoch training loop with NLL loss
for epoch in range(1):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
```

Finally, we compute the predictive accuracy on the test set. The test accuracy looks pretty bad, basically coin-toss accuracy, which means our transformer model hasn't really learned anything useful (or that something has gone wrong). As mentioned before, the model should probably train for longer to properly fine-tune it on our dataset. In addition, the contextual embeddings which the pretrained model has learned may not be useful for our application.

```python
# Set model back to evaluation mode
model.eval()

# Count the number of correctly predicted labels on the test set
correct = 0
for batch in test_loader:
    input_ids = batch["input_ids"]
    label = batch["labels"]
    outputs = model(input_ids, labels=label)
    pred = torch.max(outputs.logits, 1)[1]
    correct += int(pred == label)
    
# Compute and print the test set accuracy
tf_acc = int(correct) / int(len(test_labels))
print(f"Transformer accuracy: {tf_acc:.4f}")
```

### Log Results

---

Lastly, we log the three model accuracies to a text file.

```python
log_results(
    results={
        "Graph Neural Network": gcn_acc,
        "Random Forest": rf_acc,
        "Transformer": tf_acc,
    } 
)
```


