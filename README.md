# Face & Feature Detector

Simple client-side face, eye, nose, and mouth detection using OpenCV.js.

## Files

- `index.html` — static web UI for image upload and webcam detection
- `static/detect.js` — OpenCV.js browser detector script
- `haarcascade_frontalface_default.xml`
- `haarcascade_eye.xml`
- `haarcascade_mcs_nose.xml`
- `haarcascade_mcs_mouth.xml`

## Run locally

1. Start a simple static server from the repo root:

```bash
python3 -m http.server 8000
```

2. Open in your browser:

```text
http://localhost:8000/
```

3. Allow camera access for live detection, or upload an image.


## Copyright / Credits

This project uses Haar cascade XML files from OpenCV and its contributors.

- `haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`: Intel License Agreement for Open Source Computer Vision Library.
- `haarcascade_mcs_nose.xml` and `haarcascade_mcs_mouth.xml`: Contributors License Agreement (Modesto Castrillon-Santana).

The XML files are redistributed here as part of the project and retain their original license requirements.
