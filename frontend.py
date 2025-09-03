import streamlit as st
import tempfile
import subprocess
import sys
from pathlib import Path
import pandas as pd

# ---------- PAGE CONFIG ---------- #
st.set_page_config(
    page_title="DataScribe",
    page_icon="📊",
    layout="wide"
)

# ---------- CUSTOM STYLING ---------- #
st.markdown(
    """
    <style>
    .main-title { text-align: center; font-size: 2.8em; font-weight: 700; color: #2C3E50; }
    .subtitle { text-align: center; font-size: 1.2em; color: #555; margin-bottom: 30px; }
    .stDownloadButton button { background-color: #2C3E50; color: white; font-weight: 600; border-radius: 8px; padding: 0.6em 1.2em; }
    .stDownloadButton button:hover { background-color: #34495E; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HEADER ---------- #
st.markdown("<h1 class='main-title'>📊 DataScribe</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a CSV and generate a polished PDF EDA report automatically</p>", unsafe_allow_html=True)

# ---------- UPLOAD SECTION ---------- #
uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"], accept_multiple_files=False)

if uploaded is not None:
    # Preview dataset info
    df = pd.read_csv(uploaded, nrows=500)
    st.success(f"✅ File `{uploaded.name}` uploaded successfully!")

    with st.expander("🔍 Quick Dataset Preview", expanded=True):
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        st.dataframe(df.head(10), use_container_width=True)

    # ---------- PROCESS PDF ---------- #
    with tempfile.TemporaryDirectory() as workdir:
        in_path = Path(workdir) / uploaded.name
        out_path = Path(workdir) / "report.pdf"

        with open(in_path, "wb") as f:
            f.write(uploaded.getbuffer())

        python_exe = sys.executable
        predict_path = Path(__file__).parent / "predict.py"
        cmd = [python_exe, str(predict_path), "--input", str(in_path), "--output", str(out_path)]

        st.info("⏳ Running analysis... This may take a few seconds.")
        try:
            # Run predict.py without immediately raising exceptions
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Show stdout and stderr for debugging
            if result.stdout.strip():
                st.text("=== STDOUT ===\n" + result.stdout)
            if result.stderr.strip():
                st.text("=== STDERR ===\n" + result.stderr)

            # Check if PDF was created
            if result.returncode != 0:
                st.error("❌ Analysis failed (non-zero exit code). Check logs above.")
                st.stop()
            elif not out_path.exists():
                st.error("⚠️ PDF report was not created. Check logs above.")
                st.stop()
            else:
                st.success("🎉 Analysis complete — report generated.")

        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

        # ---------- DOWNLOAD PDF ---------- #
        pdf_bytes = out_path.read_bytes()
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name="DataScribe_Report.pdf",
            mime="application/pdf"
        )
        st.balloons()
