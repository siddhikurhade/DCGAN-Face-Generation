import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Generator, Discriminator, LATENT_DIM, initialize_weights


def main():
    parser = argparse.ArgumentParser(description="Train a DCGAN on CelebA images.")
    parser.add_argument("--data", default="data/celeba/img_align_celeba",
                        help="Folder containing face image files.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--sample-every", type=int, default=1)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("CPU training may be very slow. Google Colab GPU is recommended.")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_path.resolve()}\n"
            "Put CelebA images in this folder or pass --data PATH_TO_IMAGES."
        )

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(
        root=str(data_path.parent),
        transform=transform
    ) if data_path.name == "img_align_celeba" else None

    # ImageFolder expects class subfolders. For a flat CelebA image folder,
    # wrap it in a single class directory or use the helper folder below.
    if dataset is None:
        raise ValueError(
            "For simplicity, ImageFolder expects this structure:\n"
            "data/celeba/img_align_celeba/class_name/*.jpg\n"
            "Move images into data/celeba/img_align_celeba/celeba/ or use the "
            "recommended Colab notebook instructions in README.md."
        )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    if len(loader) == 0:
        raise ValueError("No full batches found. Add images or reduce --batch-size.")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    out = Path("outputs")
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    history = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        g_total, d_total, batches = 0.0, 0.0, 0

        for step, (real, _) in enumerate(loader, start=1):
            real = real.to(device)
            n = real.size(0)
            real_labels = torch.ones(n, device=device)
            fake_labels = torch.zeros(n, device=device)

            # Train discriminator on real and generated images.
            optimizer_d.zero_grad(set_to_none=True)
            pred_real = discriminator(real)
            loss_real = criterion(pred_real, real_labels)
            noise = torch.randn(n, LATENT_DIM, 1, 1, device=device)
            fake = generator(noise)
            pred_fake = discriminator(fake.detach())
            loss_fake = criterion(pred_fake, fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Train generator to make the discriminator classify fakes as real.
            optimizer_g.zero_grad(set_to_none=True)
            pred_fake_for_g = discriminator(fake)
            loss_g = criterion(pred_fake_for_g, real_labels)
            loss_g.backward()
            optimizer_g.step()

            g_total += loss_g.item()
            d_total += loss_d.item()
            batches += 1
            if step % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(loader)}] "
                      f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")

        avg_g = g_total / batches
        avg_d = d_total / batches
        history.append((epoch, avg_g, avg_d))
        print(f"Epoch {epoch}: Generator loss={avg_g:.4f}, Discriminator loss={avg_d:.4f}")

        if epoch % args.sample_every == 0:
            generator.eval()
            with torch.no_grad():
                images = generator(fixed_noise).cpu()
            utils.save_image(images, samples_dir / f"epoch_{epoch:03d}.png",
                             normalize=True, nrow=8)
            generator.train()

        torch.save(generator.state_dict(), out / "generator.pth")
        torch.save(discriminator.state_dict(), out / "discriminator.pth")
        torch.save({
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }, out / "checkpoint.pth")

        with open(out / "losses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "generator_loss", "discriminator_loss"])
            writer.writerows(history)

        plt.figure(figsize=(8, 5))
        plt.plot([x[0] for x in history], [x[1] for x in history], label="Generator")
        plt.plot([x[0] for x in history], [x[2] for x in history], label="Discriminator")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy loss")
        plt.title("DCGAN Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "loss_curve.png")
        plt.close()

    print("Training finished. Generator saved to outputs/generator.pth")
    print("Generated samples: outputs/samples/")
    print("Loss graph: outputs/loss_curve.png")


if __name__ == "__main__":
    main()
