"""Generic training loop shared by both approaches.

This file does not know whether it is training the unified network or one
Binary Relevance classifier. It just trains whatever model it receives.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

import config


def train_model(model, train_ds, val_ds=None, label_idx=None):
    """Trains `model` and returns it.

    Args:
        model: a CNN (untrained, random weights).
        train_ds: dataset whose items are (image, label_vector).
        val_ds: optional dataset used only to monitor overfitting.
        label_idx: None  -> use the full label vector (unified approach).
                   j     -> use only column j (one Binary Relevance classifier).
    """
    # Move the network's weights to the GPU (Colab) or leave them on the CPU.
    model = model.to(config.DEVICE)

    # DataLoader serves the dataset in batches. shuffle=True reorders the
    # images every epoch so the network does not learn the order.
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = (DataLoader(val_ds, batch_size=config.BATCH_SIZE)
                  if val_ds is not None else None)

    # Loss: measures how wrong the predictions are (a single number).
    # It applies the sigmoid internally, which is why the CNN outputs logits.
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer: the algorithm that adjusts the weights to reduce the loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):

        # ---------------- training pass ----------------
        model.train()                 # enables Dropout
        total_loss = 0.0

        for x, y in train_loader:     # one batch at a time
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            if label_idx is not None:
                # Keep only the column of this radical. The slice j:j+1
                # (instead of j) keeps the shape (batch, 1), matching the
                # model output.
                y = y[:, label_idx:label_idx + 1]

            optimizer.zero_grad()     # 1. clear gradients from the previous batch
            logits = model(x)         # 2. forward: images -> scores
            loss = loss_fn(logits, y) # 3. compare scores with the true labels
            if config.DEBUG and epoch == 0:
              print(f"  x{tuple(x.shape)}  y{tuple(y.shape)}  "
                    f"logits{tuple(logits.shape)}  loss={loss.item():.4f}")
            loss.backward()           # 4. backward: how much did each weight contribute to the error?
            optimizer.step()          # 5. adjust the weights a little

            total_loss += loss.item() * len(x)   # .item() turns the tensor into a float

        train_loss = total_loss / len(train_ds)

        # ---------------- validation pass ----------------
        message = f"Epoch {epoch + 1}/{config.EPOCHS}  train_loss={train_loss:.4f}"

        if val_loader is not None:
            model.eval()              # disables Dropout
            val_loss = 0.0
            with torch.no_grad():     # no learning here, only measuring
                for x, y in val_loader:
                    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                    if label_idx is not None:
                        y = y[:, label_idx:label_idx + 1]
                    val_loss += loss_fn(model(x), y).item() * len(x)
            message += f"  val_loss={val_loss / len(val_ds):.4f}"

        print(message)

    return model