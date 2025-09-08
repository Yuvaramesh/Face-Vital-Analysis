import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
from scipy.fft import fft
from datetime import datetime
import io
import json

# Optional PDF
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

st.set_page_config(page_title="Face Vital Monitor", layout="wide")

# ---------------- INIT ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

ppg_signal = deque(maxlen=900)
timestamps = deque(maxlen=900)

# histories
hr_values, br_values, hrv_values = deque(maxlen=300), deque(maxlen=300), deque(maxlen=300)
stress_values, para_values, wellness_values = deque(maxlen=300), deque(maxlen=300), deque(maxlen=300)
bp_sys_values, bp_dia_values = deque(maxlen=300), deque(maxlen=300)

results = {
    'heart_rate': 0, 'breathing_rate': 0, 'hrv': 0,
    'stress_index': 0, 'parasympathetic': 0, 'wellness_score': 0,
    'blood_pressure_sys': 0, 'blood_pressure_dia': 0
}

session_data = {"measurements": [], "raw_ppg_data": [], "timestamps_data": []}

# ---------------- FUNCTIONS ----------------
def extract_ppg_signal(frame, landmarks):
    try:
        h, w = frame.shape[:2]
        forehead_indices = [10, 151, 9]
        roi = [frame[int(l.y*h), int(l.x*w), 1] for l in [landmarks[i] for i in forehead_indices] if l]
        return np.mean(roi) if roi else 0
    except:
        return 0

def calculate_heart_rate(signal_data, fps=30):
    if len(signal_data) < fps*8: return 0
    detrended = signal.detrend(signal_data)
    b,a = signal.butter(4, [0.8/(fps/2), 4.0/(fps/2)], btype='band')
    filtered = signal.filtfilt(b,a,detrended)
    fft_data = fft(filtered)
    freqs = np.fft.fftfreq(len(filtered), 1/fps)
    mask = (freqs>=0.8)&(freqs<=4.0)
    if np.any(mask):
        return int(freqs[mask][np.argmax(np.abs(fft_data[mask]))]*60)
    return 0

def calculate_breathing_rate(signal_data, fps=30):
    if len(signal_data) < fps*12: return 0
    b,a = signal.butter(2, [0.1/(fps/2), 0.5/(fps/2)], btype='band')
    filtered = signal.filtfilt(b,a,signal_data)
    peaks,_ = signal.find_peaks(filtered, distance=fps*2)
    return int(len(peaks)*(60/(len(signal_data)/fps)))

def calculate_hrv(signal_data, fps=30):
    if len(signal_data)<fps*15: return 0
    filtered = signal.medfilt(signal_data,5)
    peaks,_=signal.find_peaks(filtered, distance=fps//3)
    if len(peaks)<5: return 0
    intervals=np.diff(peaks)/fps*1000
    return int(np.sqrt(np.mean(np.diff(intervals)**2)))

def calculate_stress_index(hr,hrv,br):
    return round(min(1.0,max(0.0,(max(0,(hr-70)/50)+max(0,(50-hrv)/50)+max(0,(br-15)/15))/3)),2)

def calculate_parasympathetic(hrv,br):
    return int(min(100,max(0,((min(1.0,hrv/50)+(max(0,(20-br)/10)))/2*100))))

def estimate_bp(hr,hrv,stress):
    sys, dia = 120+(hr-70)*0.5+stress*10+(50-hrv)*0.2, 80+(hr-70)*0.3+stress*6+(50-hrv)*0.12
    return int(np.clip(sys,90,180)), int(np.clip(dia,60,120))

def calculate_wellness(hr,hrv,stress,para):
    hr_score=1-abs(hr-70)/50 if hr>0 else 0.5
    return int(((hr_score+min(1,hrv/50)+(1-stress)+(para/100))/4)*100)

def save_plot_as_image(fig):
    """Convert matplotlib figure into binary image data for PDF"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

def save_pdf(measurements, results,
             ppg_signal, hr_values, br_values, hrv_values,
             stress_values, para_values, wellness_values,
             bp_sys_values, bp_dia_values):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Title
    story.append(Paragraph("📑 30-Second Health Monitoring Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Metrics
    story.append(Paragraph("Final Measurements:", styles['Heading2']))
    for k, v in results.items():
        story.append(Paragraph(f"{k}: {v}", styles['Normal']))
    story.append(Spacer(1, 12))

    # ---- Trend Plots ----
    plots = [
        ("Heart Rate Trend", hr_values, "red"),
        ("Breathing Rate Trend", br_values, "blue"),
        ("HRV Trend", hrv_values, "purple"),
        ("Stress Index Trend", stress_values, "orange"),
        ("Parasympathetic Activity Trend", para_values, "green"),
        ("Wellness Score Trend", wellness_values, "teal"),
    ]

    for title, values, color in plots:
        if len(values) > 1:
            fig, ax = plt.subplots()
            ax.plot(values, color=color)
            ax.set_title(title)
            buf_img = save_plot_as_image(fig)
            story.append(RLImage(buf_img, width=5*inch, height=2*inch))
            plt.close(fig)

    # Blood Pressure Trend
    if len(bp_sys_values) > 1 and len(bp_dia_values) > 1:
        fig, ax = plt.subplots()
        ax.plot(bp_sys_values, label="Systolic", color="darkred")
        ax.plot(bp_dia_values, label="Diastolic", color="maroon")
        ax.legend()
        ax.set_title("Blood Pressure Trend")
        buf_img = save_plot_as_image(fig)
        story.append(RLImage(buf_img, width=5*inch, height=2*inch))
        plt.close(fig)

    # Raw PPG Signal
    if len(ppg_signal) > 1:
        fig, ax = plt.subplots()
        ax.plot(ppg_signal, color="black")
        ax.set_title("Raw PPG Signal")
        buf_img = save_plot_as_image(fig)
        story.append(RLImage(buf_img, width=5*inch, height=2*inch))
        plt.close(fig)

    doc.build(story)
    buf.seek(0)
    return buf

# ---------------- UI ----------------
st.title("🩺 Face Vital - Health Monitor (Streamlit)")
start = st.button("▶ Start Monitoring (30s)")
stop = st.button("⏹ Stop Monitoring")
video_placeholder, progress_placeholder = st.empty(), st.empty()
tab1,tab2=st.tabs(["📊 Metrics","📈 Raw Signals"])
metrics_box=tab1.empty(); raw_box=tab2.empty()

if "monitoring" not in st.session_state: st.session_state.monitoring=False

if start: 
    st.session_state.monitoring=True
    session_data["measurements"].clear(); session_data["raw_ppg_data"].clear()
    ppg_signal.clear(); timestamps.clear()

if stop: st.session_state.monitoring=False

# ---------------- LOOP ----------------
if st.session_state.monitoring:
    start_time = time.time()
    duration = 30  # seconds
    frame_skip = 3  # skip frames to reduce load (process 1 of every 3 frames)

    frame_count = 0
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # skip extra frames

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            ppg = extract_ppg_signal(rgb, res.multi_face_landmarks[0].landmark)
            ppg_signal.append(ppg)
            timestamps.append(time.time())
            session_data["raw_ppg_data"].append(ppg)

            # compute metrics only every 1 second
            if len(ppg_signal) > 300 and frame_count % (30 // frame_skip) == 0:
                sig = np.array(ppg_signal)
                hr = calculate_heart_rate(sig)
                br = calculate_breathing_rate(sig)
                hrv = calculate_hrv(sig)
                stress = calculate_stress_index(hr, hrv, br)
                para = calculate_parasympathetic(hrv, br)
                sys, dia = estimate_bp(hr, hrv, stress)
                wellness = calculate_wellness(hr, hrv, stress, para)

                results.update({
                    "heart_rate": hr, "breathing_rate": br, "hrv": hrv,
                    "stress_index": stress, "parasympathetic": para,
                    "blood_pressure_sys": sys, "blood_pressure_dia": dia,
                    "wellness_score": wellness
                })

                # update histories
                hr_values.append(hr); br_values.append(br); hrv_values.append(hrv)
                stress_values.append(stress); para_values.append(para); wellness_values.append(wellness)
                bp_sys_values.append(sys); bp_dia_values.append(dia)

                session_data["measurements"].append(
                    dict(time=datetime.now().isoformat(), **results)
                )

                # Update metrics UI
                metrics_box.write(results)

        # Show live video
        video_placeholder.image(rgb, channels="RGB")

        # Update progress
        elapsed = time.time() - start_time
        progress_placeholder.progress(min(100, int((elapsed / duration) * 100)))
# ---------------- Export ----------------
st.subheader("📤 Export Data")
colA,colB,colC,colD=st.columns(4)
if colD.button("Generate PDF") and PDF_AVAILABLE:
    pdf_buffer = save_pdf(
        session_data["measurements"], results,
        list(ppg_signal), list(hr_values), list(br_values), list(hrv_values),
        list(stress_values), list(para_values), list(wellness_values),
        list(bp_sys_values), list(bp_dia_values)
    )
    st.download_button(
        label="⬇ Download PDF",
        data=pdf_buffer.getvalue(),   # ✅ must call getvalue()
        file_name="report.pdf",
        mime="application/pdf"
    )
