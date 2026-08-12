// CropGuard AI — client-side interactions
// UI/UX polish layer only: no backend calls, no fake data, no altered results.
// All progress text reflects the real in-flight request; nothing is delayed
// artificially to "look" like AI is working longer than it actually is.

document.addEventListener("DOMContentLoaded", function () {

  // ============================================================
  // Toast notifications (auto-converts Bootstrap flash alerts)
  // ============================================================
  function ensureToastContainer() {
    let c = document.getElementById("cgToastContainer");
    if (!c) {
      c = document.createElement("div");
      c.id = "cgToastContainer";
      c.className = "cg-toast-container";
      document.body.appendChild(c);
    }
    return c;
  }

  window.cgToast = function (message, type) {
    type = type || "success";
    const container = ensureToastContainer();
    const el = document.createElement("div");
    el.className = "cg-toast cg-toast-" + type;
    const icon = type === "success" ? "fa-circle-check" : type === "danger" ? "fa-triangle-exclamation" : "fa-circle-info";
    el.innerHTML = '<i class="fa-solid ' + icon + '"></i><span>' + message + "</span>";
    container.appendChild(el);
    requestAnimationFrame(() => el.classList.add("cg-toast-in"));
    setTimeout(() => {
      el.classList.remove("cg-toast-in");
      setTimeout(() => el.remove(), 250);
    }, 3800);
  };

  // Convert existing server-rendered flash alerts into toasts too, so both
  // mechanisms feel consistent (the alert box still shows for anyone with
  // JS off; with JS on we also surface a toast for a more modern feel).
  document.querySelectorAll(".cg-alert[data-flash]").forEach(el => {
    const type = el.getAttribute("data-flash") || "info";
    window.cgToast(el.textContent.trim(), type === "danger" ? "danger" : type === "warning" ? "info" : "success");
  });

  // ============================================================
  // Leaf upload experience (detect.html)
  // ============================================================
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("leafImage");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const previewSize = document.getElementById("previewSize");
  const changeImageBtn = document.getElementById("changeImageBtn");
  const removeImageBtn = document.getElementById("removeImageBtn");
  const dropzoneText = document.getElementById("dropzoneText");
  const dropzoneInner = document.getElementById("dropzoneInner");
  const submitBtn = document.getElementById("submitBtn");
  const detectForm = document.getElementById("detectForm");

  const cameraFallbackInput = document.getElementById("cameraFallbackInput");
  const galleryInput = document.getElementById("galleryInput");
  const takePhotoBtn = document.getElementById("takePhotoBtn");
  const chooseGalleryBtn = document.getElementById("chooseGalleryBtn");

  // ---- Live camera modal (getUserMedia) ----
  const cameraModal = document.getElementById("cameraModal");
  const cameraVideo = document.getElementById("cameraVideo");
  const cameraCanvas = document.getElementById("cameraCanvas");
  const cameraShutterBtn = document.getElementById("cameraShutterBtn");
  const cameraCancelBtn = document.getElementById("cameraCancelBtn");
  const cameraError = document.getElementById("cameraError");
  let cameraStream = null;

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
    }
    if (cameraModal) cameraModal.classList.remove("cg-open");
  }

  async function openCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      // Camera API unsupported (old browser) -- fall back to OS picker.
      if (cameraFallbackInput) cameraFallbackInput.click();
      return;
    }
    if (cameraError) cameraError.style.display = "none";
    cameraModal.classList.add("cg-open");
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false
      });
      cameraVideo.srcObject = cameraStream;
    } catch (err) {
      // Permission denied, no camera, or insecure context -- fall back to
      // the OS file picker rather than leaving the user stuck.
      stopCamera();
      if (cameraFallbackInput) {
        cameraFallbackInput.click();
      } else if (window.cgToast) {
        window.cgToast("Couldn't access the camera. Please allow camera permission or choose from gallery instead.", "danger");
      }
    }
  }

  function capturePhoto() {
    if (!cameraVideo.videoWidth) return;
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    const ctx = cameraCanvas.getContext("2d");
    ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
    cameraCanvas.toBlob(blob => {
      if (!blob) return;
      const file = new File([blob], "leaf-photo-" + Date.now() + ".jpg", { type: "image/jpeg" });
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      handlePreview(file);
      stopCamera();
    }, "image/jpeg", 0.92);
  }

  if (takePhotoBtn) takePhotoBtn.addEventListener("click", openCamera);
  if (cameraCancelBtn) cameraCancelBtn.addEventListener("click", stopCamera);
  if (cameraShutterBtn) cameraShutterBtn.addEventListener("click", capturePhoto);
  if (cameraFallbackInput) {
    cameraFallbackInput.addEventListener("change", () => {
      if (!cameraFallbackInput.files.length) return;
      const dt = new DataTransfer();
      dt.items.add(cameraFallbackInput.files[0]);
      fileInput.files = dt.files;
      handlePreview(cameraFallbackInput.files[0]);
    });
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function handlePreview(file) {
    if (previewName) previewName.textContent = file.name || "Photo captured";
    if (previewSize) previewSize.textContent = file.size ? formatBytes(file.size) : "";
    const reader = new FileReader();
    reader.onload = e => {
      if (previewImg) previewImg.src = e.target.result;
      if (previewWrap) previewWrap.style.display = "block";
      if (dropzoneInner) dropzoneInner.style.display = "none";
      if (dropzone) dropzone.classList.add("cg-has-file");
    };
    reader.readAsDataURL(file);
    if (window.cgToast) window.cgToast("Image selected", "success");
  }

  function clearPreview() {
    if (fileInput) fileInput.value = "";
    if (previewWrap) previewWrap.style.display = "none";
    if (dropzoneInner) dropzoneInner.style.display = "";
    if (dropzone) dropzone.classList.remove("cg-has-file");
    if (dropzoneText) dropzoneText.textContent = "Drag & drop a leaf photo here";
  }

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", e => {
      // Ignore clicks on the button row / preview controls inside the dropzone
      if (e.target.closest("button") || e.target.closest("#previewWrap")) return;
      fileInput.click();
    });

    ["dragenter", "dragover"].forEach(evt =>
      dropzone.addEventListener(evt, e => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      })
    );
    ["dragleave", "drop"].forEach(evt =>
      dropzone.addEventListener(evt, e => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      })
    );
    dropzone.addEventListener("drop", e => {
      const files = e.dataTransfer.files;
      if (files.length) {
        fileInput.files = files;
        handlePreview(files[0]);
      }
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) handlePreview(fileInput.files[0]);
    });

    // "Choose from gallery" opens the normal file picker; the file gets
    // copied into the single real form field (#leafImage) via
    // DataTransfer, since only one named file input can be submitted.
    function adoptFile(sourceInput) {
      if (!sourceInput.files.length) return;
      const dt = new DataTransfer();
      dt.items.add(sourceInput.files[0]);
      fileInput.files = dt.files;
      handlePreview(sourceInput.files[0]);
    }
    if (chooseGalleryBtn && galleryInput) {
      chooseGalleryBtn.addEventListener("click", () => galleryInput.click());
      galleryInput.addEventListener("change", () => adoptFile(galleryInput));
    }
    if (changeImageBtn) {
      changeImageBtn.addEventListener("click", e => {
        e.stopPropagation();
        fileInput.click();
      });
    }
    if (removeImageBtn) {
      removeImageBtn.addEventListener("click", e => {
        e.stopPropagation();
        clearPreview();
      });
    }
  }

  // ============================================================
  // Analysis progress overlay -- shown for exactly as long as the
  // real request takes (no artificial delay). The step list advances
  // on a light timer purely as a visual heartbeat while we wait for
  // the actual server response; if the response comes back fast,
  // the browser navigates away before all steps are shown, which is
  // fine and honest -- we're not blocking or slowing anything down.
  // ============================================================
  if (detectForm && submitBtn) {
    detectForm.addEventListener("submit", e => {
      if (fileInput && !fileInput.files.length) return; // let native validation handle it
      submitBtn.disabled = true;
      submitBtn.classList.add("cg-btn-loading");

      const overlay = document.getElementById("analyzingOverlay");
      if (overlay) {
        overlay.style.display = "flex";
        const steps = overlay.querySelectorAll(".cg-step");
        let i = 0;
        steps.forEach(s => s.classList.remove("active", "done"));
        if (steps[0]) steps[0].classList.add("active");
        const interval = setInterval(() => {
          if (i >= steps.length - 1) { clearInterval(interval); return; }
          steps[i].classList.remove("active");
          steps[i].classList.add("done");
          i++;
          steps[i].classList.add("active");
        }, 700);
      } else {
        submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Analyzing leaf…';
      }
    });
  }

  // ============================================================
  // Admin dashboard charts (admin_dashboard.html)
  // ============================================================
  function showChartFallback(canvasEl, message) {
    if (!canvasEl) return;
    const wrap = document.createElement("div");
    wrap.className = "text-muted small text-center py-4";
    wrap.innerHTML = '<i class="fa-solid fa-chart-simple mb-2 d-block" style="font-size:1.4rem;opacity:.4"></i>' + message;
    canvasEl.replaceWith(wrap);
  }

  const defChartEl = document.getElementById("deficiencyChart");
  const sevChartElCheck = document.getElementById("severityChart");
  if ((defChartEl || sevChartElCheck) && !window.Chart) {
    // Chart.js failed to load (CDN blocked, offline, etc.) -- show a clear
    // message instead of a silently blank box.
    showChartFallback(defChartEl, "Chart library failed to load. Check your connection and refresh.");
    showChartFallback(sevChartElCheck, "Chart library failed to load. Check your connection and refresh.");
  }
  if (defChartEl && window.Chart && window.cgDeficiencyData) {
    new Chart(defChartEl, {
      type: "bar",
      data: {
        labels: window.cgDeficiencyData.labels,
        datasets: [{
          label: "Scans",
          data: window.cgDeficiencyData.values,
          backgroundColor: "#40916C",
          borderRadius: 6
        }]
      },
      options: {
        plugins: { legend: { display: false }, tooltip: { padding: 10, cornerRadius: 8 } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
        animation: { duration: 400 }
      }
    });
  }

  const sevChartEl = document.getElementById("severityChart");
  if (sevChartEl && window.Chart && window.cgSeverityData) {
    new Chart(sevChartEl, {
      type: "pie",
      data: {
        labels: window.cgSeverityData.labels,
        datasets: [{
          data: window.cgSeverityData.values,
          backgroundColor: ["#2D6A4F", "#D9B44A", "#A63A2E", "#95D5B2"]
        }]
      },
      options: { plugins: { legend: { position: "bottom" }, tooltip: { padding: 10, cornerRadius: 8 } }, animation: { duration: 400 } }
    });
  }

  // ============================================================
  // Results page spectral pie chart + animated confidence ring
  // ============================================================
  const specChartEl = document.getElementById("spectralPie");
  if (specChartEl && window.Chart && window.cgSpectralData) {
    new Chart(specChartEl, {
      type: "pie",
      data: {
        labels: ["Green", "Yellow", "Brown", "Purple"],
        datasets: [{
          data: window.cgSpectralData,
          backgroundColor: ["#2D6A4F", "#D9B44A", "#8C5A3C", "#7A5C8F"]
        }]
      },
      options: { plugins: { legend: { position: "bottom" }, tooltip: { padding: 10, cornerRadius: 8 } }, animation: { duration: 500 } }
    });
  }

  const confRing = document.getElementById("confidenceRing");
  if (confRing) {
    const pct = parseFloat(confRing.getAttribute("data-value") || "0");
    const circle = confRing.querySelector(".cg-ring-progress");
    if (circle) {
      const radius = circle.r.baseVal.value;
      const circumference = 2 * Math.PI * radius;
      circle.style.strokeDasharray = circumference;
      circle.style.strokeDashoffset = circumference;
      requestAnimationFrame(() => {
        circle.style.transition = "stroke-dashoffset 900ms ease-out";
        circle.style.strokeDashoffset = circumference * (1 - pct / 100);
      });
    }
  }
});
