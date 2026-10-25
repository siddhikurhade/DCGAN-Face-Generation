# DCGAN Face Generation on CelebA

Beginner-friendly Experiment No. 6 project using PyTorch and Flask.

## Project files
- `model.py`: DCGAN Generator and Discriminator.
- `train.py`: training loop, saved model weights, sample grids and loss curve.
- `app.py`: Flask website.
- `templates/index.html`: Generate Face interface.
- `data/celeba/`: place the CelebA image data here.
- `outputs/`: training outputs are saved here.

## Recommended workflow
Train in Google Colab with a GPU, then download `outputs/generator.pth` to your Windows project for Flask deployment.

### Dataset folder layout
The provided training script uses `torchvision.datasets.ImageFolder`. Put images like this:

```
data/celeba/
  img_align_celeba/
    celeba/
      000001.jpg
      000002.jpg
      ...
```

If your extracted CelebA images are directly inside `img_align_celeba`, create the `celeba` subfolder and move the JPG files into it. Keep the original image files; do not rename them individually.

### Install and train (Windows PowerShell)
Use Python 3.11 or 3.12 for the simplest PyTorch compatibility. Python 3.13 may work with recent builds, but if installation fails, use Python 3.11/3.12.

```
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py --data data/celeba/img_align_celeba --epochs 5 --batch-size 16 --workers 0
```

On an 8 GB RAM computer, start with batch size 8 or 16. CPU training can take a very long time; a Colab GPU is recommended. Use a small subset for a quick test.

### Google Colab dataset setup
1. Open https://colab.research.google.com and create a notebook.
2. Runtime > Change runtime type > select a GPU.
3. Upload this project folder or copy `model.py` and `train.py` into notebook cells/files.
4. Upload/extract CelebA so the directory layout matches above.
5. Install compatible PyTorch/torchvision if the Colab environment needs it.
6. Run:
```
!python train.py --data data/celeba/img_align_celeba --epochs 5 --batch-size 64 --workers 2
```
7. Download `outputs/generator.pth`, `outputs/loss_curve.png`, and sample images to your PC.
For Kaggle downloads, use Kaggle's official API and keep `kaggle.json` private. Do not upload it publicly.

### Run Flask (Windows)
After training, make sure `outputs/generator.pth` exists:
```
.\\.venv\\Scripts\\Activate.ps1
python app.py
```
Open http://127.0.0.1:5000 in your browser. Click **Generate Face** to create another image.

## Notes
- Five epochs are only a starting point for a lab demo, not a guarantee of realistic faces. Quality depends on dataset size, GPU time and training stability.
- `generator.pth` contains model weights, not the dataset or the Python model definition. Keep `model.py` with it.
- Generated images are synthetic and may look blurry or distorted, especially after short training.
- CelebA contains celebrity face images. Follow the dataset's terms and use synthetic outputs responsibly.
