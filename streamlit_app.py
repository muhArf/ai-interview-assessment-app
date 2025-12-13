import streamlit as st
import os
import tempfile
import json
import pandas as pd
# --- IMPOR MODUL DARI FOLDER SRC ---
from src import nonverbal_analysis 
from src import stt_module          

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="AI Interview Assessment",
    layout="wide"
)

st.title("🤖 AI Interview Assessment Tool")
st.markdown("Unggah file audio wawancara untuk analisis STT dan Non-verbal.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Pilih File Audio Wawancara (WAV, MP3, M4A)", 
    type=["wav", "mp3", "m4a", "ogg"]
)

if uploaded_file is not None:
    
    # Placeholder untuk status pemrosesan
    status_placeholder = st.empty()
    status_placeholder.info("Mempersiapkan dan memproses file...")

    # 1. Proses File Sementara
    with tempfile.TemporaryDirectory() as temp_dir:
        
        file_name = uploaded_file.name
        temp_file_path = os.path.join(temp_dir, file_name)

        # Menulis buffer file Streamlit ke file fisik sementara
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        assessment_results = {"file_name": file_name}
        
        # --- 2. RUN ASSESSMENT STAGES ---
        
        # 2.1. STT / TRANSCRIPTION ANALYSIS
        status_placeholder.info("⏳ Melakukan Speech-to-Text dan Text Cleaning...")
        stt_result = stt_module.get_transcription_data(temp_file_path)
        assessment_results["stt_transcript"] = stt_result
        if stt_result.get("success"):
            st.success("✅ STT Transcription: Complete")
        else:
            st.error(f"❌ STT Transcription: Failed. {stt_result.get('error')}")


        # 2.2. NON-VERBAL ANALYSIS (Tempo & Jeda)
        status_placeholder.info("⏳ Melakukan Analisis Non-Verbal (Tempo dan Jeda)...")
        nonverbal_data = nonverbal_analysis.analyze_audio(temp_file_path)
        assessment_results["nonverbal_cues"] = nonverbal_data
        if "error" not in nonverbal_data:
            st.success("✅ Non-Verbal Analysis: Complete")
        else:
            st.error(f"❌ Non-Verbal Analysis: Failed. {nonverbal_data.get('error')}")


        # 2.3. CONFIDENCE SCORING (Modul Anda yang akan datang)
        # status_placeholder.info("⏳ Menghitung Confidence Score dan Rubrik...")
        # assessment_results["confidence_rubric"] = your_confidence_module.calculate_score(...)
        # st.success("✅ Confidence Scoring: Complete")
        
        
        status_placeholder.empty() # Hapus placeholder status
        st.header("✨ Hasil Penilaian Wawancara")
        
        # --- 3. TAMPILKAN HASIL ---

        # Tampilan 1: Transkripsi
        st.subheader("📝 Transkripsi Bersih")
        if assessment_results["stt_transcript"].get("clean_transcript"):
            st.markdown(assessment_results["stt_transcript"]["clean_transcript"])
        else:
            st.warning("Transkripsi tidak tersedia.")
            
        
        # Tampilan 2: Non-Verbal Cues (Tabel Ringkas)
        st.subheader("🗣️ Non-Verbal Cues (Tempo & Jeda)")
        nv_data = assessment_results.get("nonverbal_cues", {})
        if "error" not in nv_data:
            df = pd.DataFrame({
                "Metric": ["Tempo", "Total Pause", "Summary"],
                "Value": [
                    nv_data.get("tempo_bpm", "N/A"),
                    nv_data.get("total_pause_seconds", "N/A"),
                    nv_data.get("qualitative_summary", "N/A")
                ]
            })
            st.table(df)
        else:
            st.error(nv_data.get("error"))

        # Tampilan 3: Full JSON Output
        st.subheader("📦 Full Assessment JSON Data")
        st.json(assessment_results)
        
        # Opsi Unduh JSON
        json_string = json.dumps(assessment_results, indent=4)
        st.download_button(
            label="Download Full Assessment JSON",
            data=json_string,
            file_name=f"{file_name}_assessment_results.json",
            mime="application/json"
        )

else:
    st.info("Silakan unggah file audio wawancara Anda untuk memulai proses penilaian.")