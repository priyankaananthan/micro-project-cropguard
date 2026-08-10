// CropGuard AI — client-side interactions

document.addEventListener("DOMContentLoaded", function () {
  // ---- Leaf upload dropzone (detect.html) ----
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("leafImage");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const dropzoneText = document.getElementById("dropzoneText");
  const submitBtn = document.getElementById("submitBtn");
  const detectForm = document.getElementById("detectForm");

  const cameraInput = document.getElementById("cameraInput");
  const galleryInput = document.getElementById("galleryInput");
  const takePhotoBtn = document.getElementById("takePhotoBtn");
  const chooseGalleryBtn = document.getElementById("chooseGalleryBtn");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

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

    // "Take a photo" / "Choose from gallery" -- each is a separate hidden
    // input (camera one has capture="environment" to open the device
    // camera directly). Whichever the user picks gets copied into the
    // single real form field (#leafImage) via DataTransfer, since only
    // one named file input can be submitted.
    function adoptFile(sourceInput) {
      if (!sourceInput.files.length) return;
      const dt = new DataTransfer();
      dt.items.add(sourceInput.files[0]);
      fileInput.files = dt.files;
      handlePreview(sourceInput.files[0]);
    }
    if (takePhotoBtn && cameraInput) {
      takePhotoBtn.addEventListener("click", () => cameraInput.click());
      cameraInput.addEventListener("change", () => adoptFile(cameraInput));
    }
    if (chooseGalleryBtn && galleryInput) {
      chooseGalleryBtn.addEventListener("click", () => galleryInput.click());
      galleryInput.addEventListener("change", () => adoptFile(galleryInput));
    }

    function handlePreview(file) {
      dropzoneText.textContent = file.name || "Photo captured";
      const reader = new FileReader();
      reader.onload = e => {
        previewImg.src = e.target.result;
        previewWrap.style.display = "block";
      };
      reader.readAsDataURL(file);
    }
  }

  if (detectForm && submitBtn) {
    detectForm.addEventListener("submit", () => {
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Analyzing leaf…';
    });
  }

  // ---- Admin dashboard charts (admin_dashboard.html) ----
  const defChartEl = document.getElementById("deficiencyChart");
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
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
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
      options: { plugins: { legend: { position: "bottom" } } }
    });
  }

  // ---- Results page spectral pie chart ----
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
      options: { plugins: { legend: { position: "bottom" } } }
    });
  }
});
