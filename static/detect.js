// detect.js — simple OpenCV.js-based detector
(function () {
  const statusEl = () => document.getElementById("status");
  const canvas = document.getElementById("canvasOutput");
  const video = document.getElementById("video");
  const fileInput = document.getElementById("fileInput");
  let stream = null;

  async function fetchAndLoadCascade(name) {
    try {
      const res = await fetch("/" + name);
      const buf = await res.arrayBuffer();
      cv.FS_createDataFile("/", name, new Uint8Array(buf), true, false, false);
      return true;
    } catch (e) {
      console.error("Failed load cascade", name, e);
      return false;
    }
  }

  async function initCascades() {
    statusEl().textContent = "Memuat model...";
    await fetchAndLoadCascade("haarcascade_frontalface_default.xml");
    await fetchAndLoadCascade("haarcascade_eye.xml");
    await fetchAndLoadCascade("haarcascade_mcs_nose.xml");
    await fetchAndLoadCascade("haarcascade_mcs_mouth.xml");
    statusEl().textContent = "Model siap.";
  }

  function drawRects(src, rects, color) {
    for (let i = 0; i < rects.size(); ++i) {
      let r = rects.get(i);
      cv.rectangle(
        src,
        new cv.Point(r.x, r.y),
        new cv.Point(r.x + r.width, r.y + r.height),
        color,
        2,
      );
    }
  }

  function detectImage() {
    try {
      let src = cv.imread(canvas);
      let gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

      let faceCascade = new cv.CascadeClassifier();
      faceCascade.load("haarcascade_frontalface_default.xml");
      let faces = new cv.RectVector();
      let msize = new cv.Size(0, 0);
      faceCascade.detectMultiScale(gray, faces, 1.3, 5, 0, msize, msize);
      drawRects(src, faces, new cv.Scalar(255, 0, 0, 255));

      let eyeCascade = new cv.CascadeClassifier();
      eyeCascade.load("haarcascade_eye.xml");
      let eyes = new cv.RectVector();
      eyeCascade.detectMultiScale(gray, eyes, 1.1, 22);
      drawRects(src, eyes, new cv.Scalar(36, 255, 12, 255));

      let noseCascade = new cv.CascadeClassifier();
      noseCascade.load("haarcascade_mcs_nose.xml");
      let noses = new cv.RectVector();
      noseCascade.detectMultiScale(gray, noses, 1.1, 22);
      drawRects(src, noses, new cv.Scalar(0, 255, 255, 255));

      let mouthCascade = new cv.CascadeClassifier();
      mouthCascade.load("haarcascade_mcs_mouth.xml");
      let mouths = new cv.RectVector();
      mouthCascade.detectMultiScale(gray, mouths, 1.1, 22);
      drawRects(src, mouths, new cv.Scalar(255, 0, 255, 255));

      cv.imshow(canvas, src);

      // cleanup
      src.delete();
      gray.delete();
      faces.delete();
      eyes.delete();
      noses.delete();
      mouths.delete();
      faceCascade.delete();
      eyeCascade.delete();
      noseCascade.delete();
      mouthCascade.delete();
    } catch (err) {
      console.error(err);
      statusEl().textContent = "Deteksi gagal: " + err;
    }
  }

  function stopCam() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      video.style.display = "none";
    }
  }

  document.getElementById("startCam").addEventListener("click", async () => {
    stopCam();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      video.style.display = "block";
      statusEl().textContent = "Webcam aktif.";
      // draw video frame to canvas continuously
      const ctx = canvas.getContext("2d");
      video.addEventListener("playing", function () {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        (function drawLoop() {
          if (!stream) return;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          requestAnimationFrame(drawLoop);
        })();
      });
    } catch (e) {
      statusEl().textContent = "Gagal buka webcam.";
      console.error(e);
    }
  });

  document.getElementById("stopCam").addEventListener("click", () => {
    stopCam();
    statusEl().textContent = "Webcam dihentikan.";
  });

  fileInput.addEventListener("change", (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
    };
    img.src = URL.createObjectURL(f);
  });

  document.getElementById("detectBtn").addEventListener("click", () => {
    detectImage();
    statusEl().textContent = "Deteksi selesai.";
  });

  // Wait for OpenCV to be ready then load cascades
  if (typeof cv === "undefined") {
    let check = setInterval(() => {
      if (typeof cv !== "undefined" && cv["onRuntimeInitialized"]) {
        clearInterval(check);
        cv["onRuntimeInitialized"] = () => {
          initCascades();
        };
      }
    }, 100);
  } else {
    cv["onRuntimeInitialized"] = () => {
      initCascades();
    };
  }
})();
