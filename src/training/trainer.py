
"""Generic trainer shared by the Unified and Binary Relevance approaches.

The Trainer is responsible for the complete training process:
batch preparation, forward passes, loss calculation, backpropagation,
weight updates, and optional validation.

It does not depend on a specific classification approach. It simply
trains the model and labels it receives.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config


class Trainer:
    """Handle the training and validation of neural network models.

    The Trainer can be used with any model compatible with the training
    process implemented here.

    It supports both classification approaches used in this project:

    - Unified: trains the model using the complete multi-label vector.
    - Binary Relevance: trains the model using one label at a time.

    Attributes:
        batch_size (int): Number of images processed in each batch.
        epochs (int): Number of complete passes through the training dataset.
        learning_rate (float): Step-size parameter used by the optimizer.
        device: Device used to perform computations (CPU or GPU).
        debug (bool): Whether to print additional training information.
    """

    def __init__(self) -> None:
        """Initialize the trainer using the project configuration."""

        self.batch_size = config.BATCH_SIZE
        self.epochs = config.EPOCHS
        self.learning_rate = config.LEARNING_RATE
        self.device = config.DEVICE
        self.debug = config.DEBUG
        self.warmup_epochs = config.WARMUP_EPOCHS
        self.warmup_start_factor = config.WARMUP_START_FACTOR

    def train(
        self,
        model: nn.Module,
        train_ds: Dataset,
        val_ds: Dataset | None = None,
        label_idx: int | None = None
    ) -> nn.Module:
        """Train a model using the provided training dataset.

        The label_idx argument determines which target labels are used:

        - None: use the complete multi-label vector (Unified).
        - An integer: use only the corresponding label column
          (Binary Relevance).

        Args:
            model (nn.Module): Neural network to train.
            train_ds (Dataset): Training dataset containing images and
                their complete multi-label vectors.
            val_ds (Dataset | None): Optional validation dataset used
                to monitor the model's performance during training.
            label_idx (int | None): Index of the radical to train on.
                If None, all radical labels are used.

        Returns:
            nn.Module: The trained model with its updated parameters.
        """

        # Move the model to the selected device (CPU or GPU).
        model = model.to(self.device)

        # Create DataLoaders to provide the data in batches.
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = (
            DataLoader(val_ds, batch_size=self.batch_size)
            if val_ds is not None else None
        )

        # Define the loss function and optimizer.
        loss_fn = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate
        )

        # The scheduler changes the optimizer's learning rate over time.
        # LinearLR grows it linearly from LEARNING_RATE * start_factor up to
        # LEARNING_RATE during `warmup_epochs` epochs, then keeps it constant.
        scheduler = None
        if self.warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )

        # Repeat the training process for the configured number of epochs.
        for epoch in range(self.epochs):

            # ---------------- training pass ----------------

            model.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # For Binary Relevance, keep only the target column
                # corresponding to the radical being trained.
                if label_idx is not None:
                    labels = labels[:, label_idx:label_idx + 1]

                # 1. Clear gradients from the previous batch.
                optimizer.zero_grad()

                # 2. Forward pass: obtain the model's raw output scores.
                logits = model(images)

                # 3. Calculate the difference between predictions and labels.
                loss = loss_fn(logits, labels)

                # Print tensor shapes and loss for debugging.
                if self.debug and epoch == 0:
                    print(
                        f"  images{tuple(images.shape)}  "
                        f"labels{tuple(labels.shape)}  "
                        f"logits{tuple(logits.shape)}  "
                        f"loss={loss.item():.4f}"
                    )

                # 4. Backpropagation: calculate gradients.
                loss.backward()

                # 5. Update the model's parameters.
                optimizer.step()

                # Accumulate the loss weighted by the batch size.
                total_loss += loss.item() * len(images)

            # Calculate the average training loss for this epoch.
            train_loss = total_loss / len(train_ds)

            # Prepare the epoch summary.
            message = (
                f"Epoch {epoch + 1}/{self.epochs}  "
                f"train_loss={train_loss:.4f}"
                f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            # ---------------- validation pass ----------------

            if val_loader is not None:
                model.eval()
                val_loss = 0.0

                # Disable gradient calculations during validation.
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        # Apply the same label selection used in training.
                        if label_idx is not None:
                            labels = labels[:, label_idx:label_idx + 1]

                        logits = model(images)
                        loss = loss_fn(logits, labels)

                        val_loss += loss.item() * len(images)

                # Calculate the average validation loss.
                average_val_loss = val_loss / len(val_ds)

                message += f"  val_loss={average_val_loss:.4f}"

             # Advance the schedule by one epoch
            if scheduler is not None:
                scheduler.step()

            print(message)

        return model
