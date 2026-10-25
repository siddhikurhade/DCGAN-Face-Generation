from pathlib import Path
import secrets

import torch
from flask import Flask, render_template
from torchvision.utils import save_image

from model import Generator, LATENT_DIM

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "generator.pth"
STATIC_GENERATED = BASE_DIR / "static" / "generated"
STATIC_GENERATED.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

if MODEL_PATH.exists():
    # This is a state_dict produced by this project's train.py.
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    generator.eval()
    MODEL_READY = True
else:
    MODEL_READY = False


@app.route("/", methods=["GET", "POST"])
def index():
    image_name = None
    message = None
    if not MODEL_READY:
        message = "Model not found. Train the DCGAN first and place generator.pth in outputs/."
    else:
        noise = torch.randn(1, LATENT_DIM, 1, 1, device=device)
        with torch.no_grad():
            image = generator(noise).cpu()
        image_name = f"face_{secrets.token_hex(8)}.png"
        save_image(image, STATIC_GENERATED / image_name, normalize=True)
    return render_template("index.html", image_name=image_name, message=message)


if __name__ == "__main__":
    app.run(debug=True)
