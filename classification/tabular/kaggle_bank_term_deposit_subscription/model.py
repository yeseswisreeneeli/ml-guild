from torch import nn
import torch.optim as optim
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-o", "--output_dir", help="Output directory to save model", required=True)
args = parser.parse_args()

"""DataLoader holds the given torch_dataset into to sets of feature set 
and label set in given batch size

The dataset is processed in data_prep.py"""
train_dataloader = DataLoader(train_set,batch_size,shuffle=True)
test_dataloader = DataLoader(test_set,batch_size,shuffle=True)


input_size = 12
hidden_size = 32
output_size = 1
batch_size = 16
epochs = 1

torch.manual_seed(23)

class binaryClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.Layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.Layer3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.Layer4 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Layer1(x)
        x = self.relu1(x)
        x = self.Layer2(x)
        x = self.relu2(x)
        x = self.Layer3(x)
        x = self.relu3(x)
        x = self.Layer4(x)
        x = self.sigmoid(x)
        return x

model = binaryClassificationModel().to(device)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

#training loop
print("[INFO] Training & Evaluating Model...")
for epoch in range(epochs):
    model.train()
    for batch_features, batch_labels in train_dataloader:
        outputs = model(batch_features)

        # Calculate loss
        loss = loss_fn(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Train: loss:{loss}|accuracy:{1-loss}")

    model.eval()
    with torch.inference_mode():
    for batch_features, batch_labels in test_dataloader:
        outputs = model(batch_features)

        # Calculate loss
        loss = loss_fn(outputs, batch_labels)

    print(f"Eval: loss:{loss}|accuracy:{1-loss}")



# torch.save(model, args.output_dir) #saving model

##Inference
# model_2 = torch.load("/content/model/term_depost_subscription_model.pt")
# for batch_features, batch_labels in test_dataloader:
#   raise ValueError(model_2(batch_features)[0], batch_labels[0])
