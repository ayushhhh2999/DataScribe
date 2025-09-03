package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// maxUploadSize sets a sane upper bound for CSV uploads (50 MB)
const maxUploadSize = 50 << 20

func main() {
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	http.HandleFunc("/predict", handlePredict)

	addr := ":8080"
	log.Printf("Server listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

// handlePredict accepts a multipart/form-data request with a 'file' field (CSV).
// It invokes the local Python script (predict.py) to analyze the CSV and produce a PDF.
// The PDF is streamed back to the client as application/pdf.
func handlePredict(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Use POST with multipart/form-data (field name: file)", http.StatusMethodNotAllowed)
		return
	}

	// Limit the size to avoid exhausting memory
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)

	// Parse multipart form
	if err := r.ParseMultipartForm(maxUploadSize); err != nil {
		http.Error(w, fmt.Sprintf("failed to parse form: %v", err), http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "missing 'file' field in form-data", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Create a working temp directory
	workdir, err := os.MkdirTemp("", "predict_job_*")
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create temp dir: %v", err), http.StatusInternalServerError)
		return
	}
	// Clean up temp directory after response is sent
	defer os.RemoveAll(workdir)

	// Save uploaded CSV
	inPath := filepath.Join(workdir, sanitizeFilename(header.Filename))
	outPath := filepath.Join(workdir, "report.pdf")

	inFile, err := os.Create(inPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create temp file: %v", err), http.StatusInternalServerError)
		return
	}
	defer inFile.Close()

	if _, err := io.Copy(inFile, file); err != nil {
		http.Error(w, fmt.Sprintf("failed to save uploaded file: %v", err), http.StatusInternalServerError)
		return
	}

	// Run the Python analysis
	cmd := exec.Command("python3", "predict.py", "--input", inPath, "--output", outPath)
	cmd.Dir = "." // run from current directory; ensure predict.py is colocated with this binary
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	start := time.Now()
	if err := cmd.Run(); err != nil {
		http.Error(w, fmt.Sprintf("analysis failed: %v\n%s", err, stderr.String()), http.StatusInternalServerError)
		return
	}
	log.Printf("Analysis finished in %s", time.Since(start))

	// Open and stream the resulting PDF
	report, err := os.Open(outPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to open generated PDF: %v", err), http.StatusInternalServerError)
		return
	}
	defer report.Close()

	// Set headers for file download
	w.Header().Set("Content-Type", "application/pdf")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s"`, "report.pdf"))
	w.Header().Set("Cache-Control", "no-store")

	// Stream the file efficiently
	buf := bufio.NewReader(report)
	if _, err := buf.WriteTo(w); err != nil {
		log.Printf("error streaming pdf: %v", err)
	}
}

// sanitizeFilename does minimal cleanup for an uploaded filename.
func sanitizeFilename(name string) string {
	if name == "" {
		return "upload.csv"
	}
	// Remove any path separators
	base := filepath.Base(name)
	return base
}
