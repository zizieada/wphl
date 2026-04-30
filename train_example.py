"""
train_example.py

Minimal example demonstrating pairwise training with the Weighted Pairwise
Hinge Loss (WPHL) on top of a generic IQA backbone.

This script uses a placeholder model and a synthetic dataset to keep the
example self-contained. Replace the model and dataset with your own.

To use with CKDN, swap the backbone for the CKDN encoder and adjust the
preprocessing pipeline to match the original CKDN implementation:
    https://github.com/researchmm/CKDN
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from WeightedPairwiseHinge import WeightedPairwiseHinge


# ---------------------------------------------------------------------------
# Placeholder backbone
# A minimal regression head on top of a MobileNetV2 feature extractor.
# Replace this with CKDN or your own model.
# ---------------------------------------------------------------------------

class IQABackbone(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.features = mobilenet.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x).squeeze(1)   # shape: (B,)


# ---------------------------------------------------------------------------
# Placeholder dataset
# Returns (img1, img2, mos1, mos2) pairs with random data.
# Replace this with your actual paired IQA dataset.
# ---------------------------------------------------------------------------

class SyntheticPairDataset(Dataset):
    """
    Synthetic dataset for demonstration only.

    In practice, pair each image in your dataset with several others,
    using their MOS values as y1 and y2. Only pairs where |y1 - y2|
    exceeds a minimum threshold are informative for ranking; feel free
    to filter out near-tied pairs.
    """
    def __init__(self, n_pairs=512, img_size=224):
        self.n_pairs  = n_pairs
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        # Random images (replace with real image loading)
        img1 = torch.rand(3, self.img_size, self.img_size)
        img2 = torch.rand(3, self.img_size, self.img_size)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        # Random MOS values in [1, 5] (replace with real labels)
        mos1 = torch.empty(1).uniform_(1.0, 5.0).item()
        mos2 = torch.empty(1).uniform_(1.0, 5.0).item()

        return img1, img2, mos1, mos2


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    # WPHL hyperparameters
    margin: float = 0.1,
    p: float = 1.0,
    max_loss_coeff: float = 0.0,   # set > 0 to penalise worst-case pairs
    # Optional: anneal max_loss_coeff linearly to a target value by the end
    max_loss_coeff_final: float = 0.2,
    anneal_max_loss: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Model
    model = IQABackbone().to(device)

    # Loss
    criterion = WeightedPairwiseHinge(
        margin=margin,
        p=p,
        initial_max_loss_coeff=max_loss_coeff,
    )

    # Data
    dataset    = SyntheticPairDataset(n_pairs=1024)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)

    # Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        # Optionally anneal max_loss_coeff
        if anneal_max_loss and num_epochs > 1:
            frac = epoch / (num_epochs - 1)
            criterion.max_loss_coeff = (
                max_loss_coeff + frac * (max_loss_coeff_final - max_loss_coeff)
            )

        model.train()
        epoch_loss = 0.0
        epoch_acc  = 0.0

        for img1, img2, mos1, mos2 in dataloader:
            img1  = img1.to(device)
            img2  = img2.to(device)
            mos1  = mos1.float().to(device)
            mos2  = mos2.float().to(device)

            # Ranking target: +1 if image 1 has higher quality, -1 otherwise
            # Pairs where mos1 == mos2 contribute zero loss regardless.
            target = torch.sign(mos1 - mos2).to(device)

            # Forward pass
            o1 = model(img1)
            o2 = model(img2)

            loss, acc = criterion(o1, o2, mos1, mos2, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc  += acc.item()

        n = len(dataloader)
        print(
            f"Epoch [{epoch+1:>3}/{num_epochs}]  "
            f"Loss: {epoch_loss/n:.4f}  "
            f"Acc: {epoch_acc/n:.4f}  "
            f"λ(max): {criterion.max_loss_coeff:.3f}"
        )

    # Save checkpoint
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_config": {
            "margin": margin,
            "p": p,
            "max_loss_coeff": criterion.max_loss_coeff,
        },
    }, "checkpoint.pth")
    print("Saved checkpoint.pth")


if __name__ == "__main__":
    train(
        num_epochs=10,
        batch_size=32,
        lr=1e-4,
        margin=0.1,
        p=1.0,
        anneal_max_loss=True,
        max_loss_coeff=0.0,
        max_loss_coeff_final=0.2,
    )
