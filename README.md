# Invoice-AI — Bill & Invoice Text Extractor

A web app that scans photos of invoices and bills and extracts their text using
optical character recognition. It also auto-parses the fields that matter most —
**invoice number, date, and total** — and lets you download the result as a `.txt`
file. Built with **Flask** (web server) and **EasyOCR** (OCR engine), with an
**OpenCV** preprocessing pass for better accuracy on real-world phone photos.

**OCR engine:** [EasyOCR](https://github.com/JaidedAI/EasyOCR) (CRAFT text
detector + a Latin recognition model)

## Prerequisites

- Python 3
- Flask
- EasyOCR
- PyTorch (`torch==2.1.0`, `torchvision==0.16.0`)
- OpenCV (`opencv-python`)
- NumPy (`numpy==1.26.1`)
- Pillow

## Project Structure

The whole app lives in one folder with exactly four sub-folders:

```
invoice-ai/
├── app.py            # Flask backend + OCR pipeline
├── requirements.txt  # Python dependencies
├── templates/        # index.html — the web page
├── static/           # style.css + script.js — the interface
├── uploads/          # images you submit (created automatically)
└── outputs/          # extracted-text results, one .txt per scan
```

## How to Run

**Get the code first.** The commands below assume you already have the
`invoice-ai/` project folder on your machine. Nothing here downloads it for you —
so obtain it one of these ways:

- Clone it from GitHub (if you've hosted it):
  ```
  git clone https://github.com/<your-username>/invoice-ai.git
  cd invoice-ai
  ```
- Or copy the whole `invoice-ai/` folder over manually.

Either way, make sure the folder contains all of these before continuing:

```
invoice-ai/
├── app.py
├── requirements.txt
├── templates/index.html
└── static/style.css, static/script.js
```

The `uploads/` and `outputs/` folders are created automatically on first run, so
they don't need to exist yet.

Install all prerequisites. The easiest way is with the included requirements file:

```
cd invoice-ai
pip install -r requirements.txt
```

Start the web server:

```
python3 app.py
```

On the **first scan only**, EasyOCR downloads its detection and recognition
models (~100 MB) into `~/.EasyOCR/`. This is a one-time step — later scans reuse
the cached models.

Open the site in a browser:

- On the same machine: `http://localhost:5000`
- From another device on the same network: `http://<this-machine-ip>:5000`
  (find the IP with `hostname -I`)

Upload an invoice or bill image (drag-and-drop or click). If the photo is
sideways, pick how to turn it from the dropdown:

- **Auto-detect** — tries all four rotations and keeps the most confident
  reading (works hands-off, but slower because it reads the image four times).
- **Turn left / Turn right / Upside down / Already upright** — applies a fixed
  rotation instantly (faster when you can see which way the image is turned).

Press **Scan invoice**. When it finishes you'll see:

- summary chips: lines found, average OCR confidence, and any parsed
  invoice number / date / total
- the full extracted text
- a **Download .txt** button (the same text is also saved in `outputs/`)

## How It Works

Each scan runs through this pipeline in `app.py`:

1. **Preprocess (OpenCV):** convert to grayscale, resize into a size band
   (upscale tiny images, shrink huge ones so scanning stays fast), even out
   lighting with CLAHE contrast, denoise, correct orientation, and deskew small
   tilts.
2. **OCR (EasyOCR):** run text detection + recognition with a greedy decoder
   tuned for speed on CPU.
3. **Reading-order reconstruction:** group detected boxes into rows
   top-to-bottom and left-to-right, so an item and its price stay on the same
   line and field parsing is more reliable.
4. **Field parsing:** pull invoice number, date, and total with
   OCR-error-tolerant regular expressions.

## Notes & Tips

- **Speed:** on a CPU-only machine (no CUDA), a scan takes roughly 8–13 seconds;
  most of that is the text-detection network. Auto-detect rotation is ~4× slower
  because it reads the image four times — pick the rotation manually when you
  know it.
- **Accuracy:** clear, well-lit, straight-on photos work best. Cropping to just
  the receipt before uploading helps.
- **GPU:** the app runs EasyOCR on CPU (`gpu=False` in `app.py`). On a machine
  with a working CUDA PyTorch install, switching this to `gpu=True` gives a large
  speedup.
- The uploaded image and the extracted `.txt` are kept in `uploads/` and
  `outputs/`; delete them anytime to clean up.

## Acknowledgments

- OCR powered by [EasyOCR](https://github.com/JaidedAI/EasyOCR) by JaidedAI
- Web server built with [Flask](https://flask.palletsprojects.com/)
- Image preprocessing with [OpenCV](https://opencv.org/)
