const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const statusDiv = document.getElementById("status");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  statusDiv.textContent = "Uploading and processing...";

  const file = fileInput.files[0];
  if (!file) {
    statusDiv.textContent = "Please select a CSV file.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://localhost:8080/predict", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errText = await response.text();
      statusDiv.textContent = "Error: " + errText;
      return;
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "report.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);

    statusDiv.textContent = "PDF generated successfully!";
  } catch (err) {
    statusDiv.textContent = "Error: " + err.message;
  }
});
