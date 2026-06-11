// detect.js — simple OpenCV.js-based detector
(function () {
  const statusEl = () => document.getElementById("status");
  const canvas = document.getElementById("canvasOutput");
  const video = document.getElementById("video");
  const fileInput = document.getElementById("fileInput");
  let stream = null;
  let videoLoopRunning = false;
  let isDrawing = false;
  let detectIntervalId = null;
  let faceCascade = null,
    eyeCascade = null,
    noseCascade = null,
    mouthCascade = null;
  let lastRawImageData = null;

  // create overlay canvas to draw boxes/text (keeps main canvas raw)
  const overlay = document.createElement("canvas");
  overlay.id = "overlayCanvas";
  overlay.style.position = "absolute";
  overlay.style.top = "0";
  overlay.style.left = "0";
  overlay.style.pointerEvents = "none";
  overlay.style.maxWidth = "100%";
  // ensure parent is positioned
  if (canvas.parentNode)
    canvas.parentNode.style.position =
      canvas.parentNode.style.position || "relative";
  canvas.parentNode.appendChild(overlay);
  const overlayCtx = overlay.getContext("2d", { willReadFrequently: true });

  async function fetchAndLoadCascade(name) {
    const candidates = ["./" + name, "/" + name, "/static/" + name, name];
    for (const path of candidates) {
      try {
        const res = await fetch(path);
        if (!res.ok) {
          console.debug("cascade not found at", path, "status", res.status);
          continue;
        }
        const buf = await res.arrayBuffer();
        cv.FS_createDataFile(
          "/",
          name,
          new Uint8Array(buf),
          true,
          false,
          false,
        );
        console.log("Loaded cascade", name, "from", path);
        return true;
      } catch (e) {
        console.debug("fetch failed for", path, e);
        continue;
      }
    }
    console.error("Failed to load cascade from any candidate path:", name);
    return false;
  }

  // initCascades is defined later after cv runtime is ready

  // drawRects removed: overlay drawing handled in detectImage to keep main canvas unmodified

  function detectImage() {
    if (!window.cascadesLoaded) {
      statusEl().textContent = "Model not ready — please wait...";
      return;
    }
    try {
      let src = cv.imread(canvas);
      let gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

      // use preloaded cascade classifiers and draw results on overlay
      overlay.width = canvas.width;
      overlay.height = canvas.height;
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      overlayCtx.lineWidth = 2;
      overlayCtx.font = "16px sans-serif";

      let faces = new cv.RectVector();
      let msize = new cv.Size(0, 0);
      if (faceCascade)
        faceCascade.detectMultiScale(gray, faces, 1.3, 5, 0, msize, msize);
      for (let i = 0; i < faces.size(); ++i) {
        const r = faces.get(i);
        overlayCtx.strokeStyle = "rgba(255,0,0,0.9)";
        overlayCtx.fillStyle = "rgba(255,0,0,0.9)";
        overlayCtx.strokeRect(r.x, r.y, r.width, r.height);
        overlayCtx.fillText("Wajah", r.x, Math.max(12, r.y - 6));
      }

      let eyes = new cv.RectVector();
      if (eyeCascade) eyeCascade.detectMultiScale(gray, eyes, 1.1, 22);
      for (let i = 0; i < eyes.size(); ++i) {
        const r = eyes.get(i);
        overlayCtx.strokeStyle = "rgba(36,255,12,0.9)";
        overlayCtx.fillStyle = "rgba(36,255,12,0.9)";
        overlayCtx.strokeRect(r.x, r.y, r.width, r.height);
        overlayCtx.fillText("Mata", r.x, Math.max(12, r.y - 6));
      }

      let noses = new cv.RectVector();
      if (noseCascade) noseCascade.detectMultiScale(gray, noses, 1.1, 22);
      for (let i = 0; i < noses.size(); ++i) {
        const r = noses.get(i);
        overlayCtx.strokeStyle = "rgba(0,255,255,0.9)";
        overlayCtx.fillStyle = "rgba(0,255,255,0.9)";
        overlayCtx.strokeRect(r.x, r.y, r.width, r.height);
        overlayCtx.fillText("Hidung", r.x, Math.max(12, r.y - 6));
      }

      let mouths = new cv.RectVector();
      if (mouthCascade) mouthCascade.detectMultiScale(gray, mouths, 1.1, 22);
      for (let i = 0; i < mouths.size(); ++i) {
        const r = mouths.get(i);
        overlayCtx.strokeStyle = "rgba(255,0,255,0.9)";
        overlayCtx.fillStyle = "rgba(255,0,255,0.9)";
        overlayCtx.strokeRect(r.x, r.y, r.width, r.height);
        overlayCtx.fillText("Mulut", r.x, Math.max(12, r.y - 6));
      }

      // cleanup small mats
      src.delete();
      gray.delete();
      faces.delete();
      eyes.delete();
      noses.delete();
      mouths.delete();
    } catch (err) {
      console.error(err);
      statusEl().textContent = "Detection failed: " + err;
    }
  }

  function stopCam() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      // ensure draw loop stops and canvas remains
      video.style.display = "none";
      if (detectIntervalId) {
        clearInterval(detectIntervalId);
        detectIntervalId = null;
      }
      isDrawing = false;
      // restore last raw frame (remove overlays)
      try {
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (lastRawImageData) ctx.putImageData(lastRawImageData, 0, 0);
      } catch (e) {
        /* ignore */
      }
      try {
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      } catch (e) {
        /* ignore */
      }
    }
  }

  document.getElementById("startCam").addEventListener("click", async () => {
    stopCam();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      // keep the raw video hidden and draw only to canvas (avoids duplicate frames)
      video.style.display = "none";
      statusEl().textContent = "Webcam active.";
      // start a single draw loop (prevent duplicates)
      startVideoLoop();
    } catch (e) {
      statusEl().textContent = "Failed to open webcam.";
      console.error(e);
    }
  });

  function startVideoLoop() {
    if (videoLoopRunning) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    videoLoopRunning = true;
    (function drawLoop() {
      if (!stream) {
        videoLoopRunning = false;
        return;
      }
      // ensure canvas matches video size
      if (video.videoWidth && video.videoHeight) {
        if (
          canvas.width !== video.videoWidth ||
          canvas.height !== video.videoHeight
        ) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          // position overlay to match canvas
          try {
            overlay.style.left = canvas.offsetLeft + "px";
            overlay.style.top = canvas.offsetTop + "px";
            overlay.width = canvas.width;
            overlay.height = canvas.height;
          } catch (e) {}
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        try {
          lastRawImageData = ctx.getImageData(
            0,
            0,
            canvas.width,
            canvas.height,
          );
        } catch (e) {
          /* ignore read errors */
        }
      }
      requestAnimationFrame(drawLoop);
    })();
    // start live detection loop if cascades are loaded
    if (!detectIntervalId) {
      detectIntervalId = setInterval(() => {
        if (window.cascadesLoaded) detectImage();
      }, 500);
    }
    // run one immediate detection when webcam starts (if model loaded)
    if (window.cascadesLoaded) detectImage();
  }

  document.getElementById("stopCam").addEventListener("click", () => {
    stopCam();
    statusEl().textContent = "Webcam stopped.";
  });

  fileInput.addEventListener("change", (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      try {
        overlay.width = canvas.width;
        overlay.height = canvas.height;
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      } catch (e) {}
    };
    img.src = URL.createObjectURL(f);
  });

  document.getElementById("detectBtn").addEventListener("click", () => {
    detectImage();
    statusEl().textContent = "Detection complete.";
  });

  // Wait for OpenCV to be ready then load cascades
  (function waitCv() {
    const check = setInterval(() => {
      if (window.cv && typeof cv.onRuntimeInitialized !== "undefined") {
        clearInterval(check);
        cv.onRuntimeInitialized = () => {
          initCascades();
        };
      }
    }, 100);
  })();

  // flag when cascades are loaded
  window.cascadesLoaded = false;

  async function initCascades() {
    statusEl().textContent = "Loading model...";
    const list = [
      "haarcascade_frontalface_default.xml",
      "haarcascade_eye.xml",
      "haarcascade_mcs_nose.xml",
      "haarcascade_mcs_mouth.xml",
    ];
    for (const f of list) {
      const ok = await fetchAndLoadCascade(f);
      if (!ok) {
        statusEl().textContent =
          "Failed to load " + f + " — check deployment paths.";
        return;
      }
    }
    try {
      // create classifiers once and keep them
      faceCascade = new cv.CascadeClassifier();
      faceCascade.load("haarcascade_frontalface_default.xml");
      eyeCascade = new cv.CascadeClassifier();
      eyeCascade.load("haarcascade_eye.xml");
      noseCascade = new cv.CascadeClassifier();
      noseCascade.load("haarcascade_mcs_nose.xml");
      mouthCascade = new cv.CascadeClassifier();
      mouthCascade.load("haarcascade_mcs_mouth.xml");
      window.cascadesLoaded = true;
      statusEl().textContent = "Model ready.";
    } catch (err) {
      console.error("Failed to initialize classifiers", err);
      statusEl().textContent = "Failed to initialize classifiers.";
    }
  }
})();
