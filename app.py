
import cv2
import numpy as np
import mediapipe as mp
import time
from scipy import signal
from scipy.fft import fft
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from collections import deque
import math
import pandas as pd
from datetime import datetime
import json
import io
# Optional PDF generation - will gracefully handle if not available
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    import base64
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class FaceVitalMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Vital - Health Monitor (MediaPipe)")
        self.root.geometry("1600x1000")
        
        # Initialize MediaPipe face detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Data storage - REDUCED TO 30 SECONDS (900 frames at 30fps)
        self.ppg_signal = deque(maxlen=900)  # 30 seconds at 30fps
        self.timestamps = deque(maxlen=900)
        
        # Individual metric signals for wave display
        self.hr_values = deque(maxlen=300)      # Heart rate history
        self.br_values = deque(maxlen=300)      # Breathing rate history
        self.hrv_values = deque(maxlen=300)     # HRV history
        self.stress_values = deque(maxlen=300)  # Stress index history
        self.para_values = deque(maxlen=300)    # Parasympathetic activity history
        self.wellness_values = deque(maxlen=300) # Wellness score history
        self.bp_sys_values = deque(maxlen=300)  # Systolic BP history
        self.bp_dia_values = deque(maxlen=300)  # Diastolic BP history
        
        self.face_detected = False
        self.monitoring_active = False
        self.calculation_count = 0  # Track number of calculations performed
        
        # Session data for export
        self.session_data = {
            'start_time': None,
            'end_time': None,
            'measurements': [],
            'raw_ppg_data': [],
            'timestamps_data': []
        }
        
        # Results
        self.results = {
            'heart_rate': 0,
            'breathing_rate': 0,
            'blood_pressure_sys': 0,
            'blood_pressure_dia': 0,
            'hrv': 0,
            'stress_index': 0,
            'parasympathetic': 0,
            'wellness_score': 0
        }
        
        self.setup_ui()
        
    def setup_ui(self):
    # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Left side - Camera and controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
    
    # Video frame
        video_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding="10")
        video_frame.pack(fill=tk.BOTH, expand=True)
    
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()
    
    # Control buttons
        button_frame = ttk.Frame(video_frame)
        button_frame.pack(pady=10)
    
        self.start_btn = ttk.Button(button_frame, text="Start Monitoring (30s)", 
                               command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
    
        self.stop_btn = ttk.Button(button_frame, text="Stop Monitoring", 
                              command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
    
    # Add Export Data button that was missing
        self.export_btn = ttk.Button(button_frame, text="Export Data", 
                                command=self.export_data, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)
    
    # PDF button - properly placed in button frame
        if PDF_AVAILABLE:
            self.pdf_btn = ttk.Button(button_frame, text="Save PDF Report",
                                 command=self.generate_pdf_report, state=tk.DISABLED)
            self.pdf_btn.pack(side=tk.LEFT, padx=5)
    
    # Progress bar
        self.progress = ttk.Progressbar(video_frame, length=400, mode='determinate')
        self.progress.pack(pady=10)
    
    # Progress label
        self.progress_label = ttk.Label(video_frame, text="Ready to start 30-second monitoring")
        self.progress_label.pack(pady=5)
    
    # Right side - Results and graphs
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
    
    # Create notebook for tabs
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
    
    # Health Metrics tab with numbers and waves
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="Health Metrics & Waves")
    
    # Raw signals tab
        signals_frame = ttk.Frame(notebook)
        notebook.add(signals_frame, text="Raw Signals")
    
        self.setup_metrics_tab(metrics_frame)
        self.setup_signals_tab(signals_frame)
    
    # Start video feed
        self.update_video()

    
    def setup_metrics_tab(self, parent):
        """Setup the health metrics tab with numbers and wave displays"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Metrics with individual wave displays
        self.setup_individual_metrics(scrollable_frame)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(scrollable_frame, text="Instructions", padding="10")
        instructions_frame.pack(fill=tk.X, pady=10, padx=10)
        
        instructions_text = """
1. Sit comfortably in front of the camera
2. Ensure your face is well-lit and clearly visible
3. Look directly at the camera and minimize movement
4. Click "Start Monitoring (30s)" and wait for 30 seconds
5. Keep still during the entire measurement period
6. Measurements start calculating after 10 seconds of data
        """
        
        ttk.Label(instructions_frame, text=instructions_text, 
                 font=('Arial', 10), justify=tk.LEFT).pack(anchor=tk.W)
    
    def setup_individual_metrics(self, parent):
        """Setup individual metric displays with numbers and mini-waves"""
        metrics_data = [
            ("Heart Rate", "heart_rate", "bpm", self.hr_values, "red"),
            ("Breathing Rate", "breathing_rate", "rpm", self.br_values, "blue"),
            ("HRV", "hrv", "ms", self.hrv_values, "green"),
            ("Stress Index", "stress_index", "", self.stress_values, "orange"),
            ("Parasympathetic", "parasympathetic", "%", self.para_values, "purple"),
            ("Wellness Score", "wellness_score", "/100", self.wellness_values, "teal")
        ]
        
        self.metric_figures = {}
        self.metric_axes = {}
        self.metric_lines = {}
        self.result_labels = {}
        self.metric_canvases = {}  # Store canvas references for updating
        
        for name, key, unit, data_deque, color in metrics_data:
            # Create frame for each metric
            metric_frame = ttk.LabelFrame(parent, text=name, padding="10")
            metric_frame.pack(fill=tk.X, pady=5, padx=10)
            
            # Create horizontal layout
            content_frame = ttk.Frame(metric_frame)
            content_frame.pack(fill=tk.BOTH, expand=True)
            
            # Left side - Number display
            number_frame = ttk.Frame(content_frame)
            number_frame.pack(side=tk.LEFT, padx=10, pady=5)
            
            self.result_labels[key] = ttk.Label(
                number_frame, text=f"-- {unit}", 
                font=('Arial', 16, 'bold'), foreground=color
            )
            self.result_labels[key].pack()
            
            # Right side - Mini wave display
            wave_frame = ttk.Frame(content_frame)
            wave_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
            
            # Create mini matplotlib figure
            fig, ax = plt.subplots(figsize=(4, 1.5))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f0f0f0')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-30, 0)  # Last 30 seconds
            ax.set_ylabel('Value', fontsize=8)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_title(f'{name} Trend', fontsize=10, fontweight='bold')
            
            line, = ax.plot([], [], color=color, linewidth=2)
            
            canvas = FigureCanvasTkAgg(fig, wave_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Store references
            self.metric_figures[key] = fig
            self.metric_axes[key] = ax
            self.metric_lines[key] = line
            self.metric_canvases[key] = canvas
            
        # Combined Blood Pressure display
        bp_frame = ttk.LabelFrame(parent, text="Blood Pressure", padding="10")
        bp_frame.pack(fill=tk.X, pady=5, padx=10)
        
        bp_content = ttk.Frame(bp_frame)
        bp_content.pack(fill=tk.BOTH, expand=True)
        
        # Number display for BP
        bp_number_frame = ttk.Frame(bp_content)
        bp_number_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.result_labels["blood_pressure"] = ttk.Label(
            bp_number_frame, text="--/-- mmHg", 
            font=('Arial', 16, 'bold'), foreground='darkred'
        )
        self.result_labels["blood_pressure"].pack()
        
        # Wave display for BP
        bp_wave_frame = ttk.Frame(bp_content)
        bp_wave_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        bp_fig, bp_ax = plt.subplots(figsize=(4, 1.5))
        bp_fig.patch.set_facecolor('white')
        bp_ax.set_facecolor('#f0f0f0')
        bp_ax.grid(True, alpha=0.3)
        bp_ax.set_xlim(-30, 0)
        bp_ax.set_ylabel('BP (mmHg)', fontsize=8)
        bp_ax.set_xlabel('Time (s)', fontsize=8)
        bp_ax.tick_params(labelsize=8)
        bp_ax.set_title('Blood Pressure Trend', fontsize=10, fontweight='bold')
        
        sys_line, = bp_ax.plot([], [], 'darkred', linewidth=2, label='Systolic')
        dia_line, = bp_ax.plot([], [], 'maroon', linewidth=2, label='Diastolic')
        bp_ax.legend(fontsize=8)
        
        bp_canvas = FigureCanvasTkAgg(bp_fig, bp_wave_frame)
        bp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store BP references
        self.metric_figures["blood_pressure"] = bp_fig
        self.metric_axes["blood_pressure"] = bp_ax
        self.metric_lines["blood_pressure_sys"] = sys_line
        self.metric_lines["blood_pressure_dia"] = dia_line
        self.metric_canvases["blood_pressure"] = bp_canvas
    
    def setup_signals_tab(self, parent):
        """Setup the raw signals visualization tab"""
        # Create matplotlib figure for raw signals
        self.signals_fig, (self.signals_ax1, self.signals_ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.signals_fig.tight_layout(pad=3.0)
        
        # PPG Signal plot
        self.signals_ax1.set_title('Raw PPG Signal', fontsize=12, fontweight='bold')
        self.signals_ax1.set_ylabel('Amplitude')
        self.signals_ax1.grid(True, alpha=0.3)
        self.signals_line1, = self.signals_ax1.plot([], [], 'b-', linewidth=1)
        
        # Filtered signals plot
        self.signals_ax2.set_title('Filtered Signals (HR & Breathing)', fontsize=12, fontweight='bold')
        self.signals_ax2.set_ylabel('Amplitude')
        self.signals_ax2.set_xlabel('Time (seconds)')
        self.signals_ax2.grid(True, alpha=0.3)
        self.signals_line2, = self.signals_ax2.plot([], [], 'r-', linewidth=2, label='HR Filter')
        self.signals_line3, = self.signals_ax2.plot([], [], 'g-', linewidth=2, label='Breathing Filter')
        self.signals_ax2.legend()
        
        # Embed matplotlib in tkinter
        self.signals_canvas = FigureCanvasTkAgg(self.signals_fig, parent)
        self.signals_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start animation for raw signals only
        self.signals_anim = FuncAnimation(self.signals_fig, self.update_signals_plots, interval=100, blit=False)
    
    def update_signals_plots(self, frame):
        """Update the raw signal plots"""
        if len(self.ppg_signal) < 10:
            return self.signals_line1, self.signals_line2, self.signals_line3
        
        # Time axis (last 30 seconds)
        time_axis = np.linspace(-len(self.ppg_signal)/30, 0, len(self.ppg_signal))
        signal_array = np.array(list(self.ppg_signal))
        
        # Update raw PPG signal plot
        self.signals_line1.set_data(time_axis, signal_array)
        self.signals_ax1.relim()
        self.signals_ax1.autoscale_view()
        
        # Update filtered signals
        if len(self.ppg_signal) > 60:
            try:
                # Heart rate filter (0.8-4 Hz)
                nyquist = 15
                hr_low = 0.8 / nyquist
                hr_high = 4.0 / nyquist
                b_hr, a_hr = signal.butter(4, [hr_low, hr_high], btype='band')
                hr_filtered = signal.filtfilt(b_hr, a_hr, signal_array)
                
                # Breathing filter (0.1-0.5 Hz)
                br_low = 0.1 / nyquist
                br_high = 0.5 / nyquist
                b_br, a_br = signal.butter(2, [br_low, br_high], btype='band')
                br_filtered = signal.filtfilt(b_br, a_br, signal_array)
                
                self.signals_line2.set_data(time_axis, hr_filtered)
                self.signals_line3.set_data(time_axis, br_filtered)
                self.signals_ax2.relim()
                self.signals_ax2.autoscale_view()
            except:
                pass
        
        return self.signals_line1, self.signals_line2, self.signals_line3
    
    def update_metrics_plots(self):
        """Update individual metric wave plots - called manually, not via animation"""
        metrics_data = [
            ("heart_rate", self.hr_values),
            ("breathing_rate", self.br_values), 
            ("hrv", self.hrv_values),
            ("stress_index", self.stress_values),
            ("parasympathetic", self.para_values),
            ("wellness_score", self.wellness_values)
        ]
        
        for key, data_deque in metrics_data:
            if key in self.metric_lines and len(data_deque) > 1:
                try:
                    time_axis = np.linspace(-len(data_deque), 0, len(data_deque))
                    data_list = list(data_deque)
                    
                    self.metric_lines[key].set_data(time_axis, data_list)
                    self.metric_axes[key].relim()
                    self.metric_axes[key].autoscale_view()
                    
                    # Set appropriate Y limits based on data type
                    if key == "heart_rate":
                        self.metric_axes[key].set_ylim(50, 150)
                    elif key == "breathing_rate":
                        self.metric_axes[key].set_ylim(8, 30)
                    elif key == "hrv":
                        self.metric_axes[key].set_ylim(0, 120)
                    elif key == "stress_index":
                        self.metric_axes[key].set_ylim(0, 1)
                    elif key == "parasympathetic":
                        self.metric_axes[key].set_ylim(0, 100)
                    elif key == "wellness_score":
                        self.metric_axes[key].set_ylim(0, 100)
                    
                    # Force canvas update
                    self.metric_canvases[key].draw_idle()
                except Exception as e:
                    print(f"Error updating {key} plot: {e}")
        
        # Update blood pressure plot
        if len(self.bp_sys_values) > 1 and len(self.bp_dia_values) > 1:
            try:
                time_axis_bp = np.linspace(-len(self.bp_sys_values), 0, len(self.bp_sys_values))
                sys_list = list(self.bp_sys_values)
                dia_list = list(self.bp_dia_values)
                
                self.metric_lines["blood_pressure_sys"].set_data(time_axis_bp, sys_list)
                self.metric_lines["blood_pressure_dia"].set_data(time_axis_bp, dia_list)
                self.metric_axes["blood_pressure"].relim()
                self.metric_axes["blood_pressure"].autoscale_view()
                self.metric_axes["blood_pressure"].set_ylim(60, 200)
                
                self.metric_canvases["blood_pressure"].draw_idle()
            except Exception as e:
                print(f"Error updating blood pressure plot: {e}")
    
    def extract_ppg_signal(self, frame, landmarks):
        """Extract PPG signal from facial regions using MediaPipe landmarks"""
        try:
            h, w = frame.shape[:2]
            
            # Define ROI indices for forehead and cheek regions
            forehead_indices = [10, 151, 9, 10, 151, 9, 10, 151]
            left_cheek_indices = [116, 117, 118, 119, 120, 121]
            right_cheek_indices = [345, 346, 347, 348, 349, 350]
            
            roi_values = []
            
            for indices in [forehead_indices, left_cheek_indices, right_cheek_indices]:
                region_points = []
                for idx in indices:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * w)
                        y = int(landmarks[idx].y * h)
                        region_points.append([x, y])
                
                if len(region_points) > 2:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(region_points)], 255)
                    
                    green_channel = frame[:, :, 1]
                    roi_mean = cv2.mean(green_channel, mask)[0]
                    roi_values.append(roi_mean)
            
            if roi_values:
                ppg_value = np.mean(roi_values)
                return ppg_value
            else:
                return 0
                
        except Exception as e:
            print(f"Error extracting PPG signal: {e}")
            return 0
    
    def calculate_heart_rate(self, signal_data, timestamps, fps=30):
        """Calculate heart rate from PPG signal"""
        if len(signal_data) < fps * 8:  # Need at least 8 seconds
            return 0
        
        try:
            # Detrend signal
            detrended = signal.detrend(signal_data)
            
            # Apply bandpass filter (0.8-4 Hz for heart rate)
            nyquist = fps / 2
            low = 0.8 / nyquist
            high = 4.0 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, detrended)
            
            # FFT analysis
            fft_data = fft(filtered)
            freqs = np.fft.fftfreq(len(filtered), 1/fps)
            
            # Find peak in heart rate range
            valid_indices = (freqs >= 0.8) & (freqs <= 4.0)
            valid_fft = np.abs(fft_data[valid_indices])
            valid_freqs = freqs[valid_indices]
            
            if len(valid_fft) > 0:
                peak_idx = np.argmax(valid_fft)
                heart_rate_hz = valid_freqs[peak_idx]
                heart_rate_bpm = heart_rate_hz * 60
                return max(50, min(200, heart_rate_bpm))
            
        except Exception as e:
            print(f"Error calculating heart rate: {e}")
        
        return 0
    
    def calculate_breathing_rate(self, signal_data, fps=30):
        """Calculate breathing rate from signal variations"""
        if len(signal_data) < fps * 12:  # Need at least 12 seconds
            return 0
        
        try:
            # Low-pass filter for breathing rate (0.1-0.5 Hz)
            nyquist = fps / 2
            low = 0.1 / nyquist
            high = 0.5 / nyquist
            b, a = signal.butter(2, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, signal_data)
            
            # Count peaks
            peaks, _ = signal.find_peaks(filtered, distance=fps*2)
            breathing_rate = len(peaks) * (60 / (len(signal_data) / fps))
            
            return max(8, min(35, breathing_rate))
            
        except Exception as e:
            print(f"Error calculating breathing rate: {e}")
            return 0
    
    def estimate_blood_pressure(self, heart_rate, hrv, stress_index):
        """Estimate blood pressure using correlation with HR, HRV, and stress"""
        try:
            base_sys = 120
            base_dia = 80
            
            hr_factor = (heart_rate - 70) * 0.5
            stress_factor = stress_index * 10
            hrv_factor = (50 - hrv) * 0.2
            
            sys_bp = base_sys + hr_factor + stress_factor + hrv_factor
            dia_bp = base_dia + hr_factor * 0.6 + stress_factor * 0.6 + hrv_factor * 0.6
            
            sys_bp = max(90, min(180, sys_bp))
            dia_bp = max(60, min(120, dia_bp))
            
            return int(sys_bp), int(dia_bp)
            
        except Exception as e:
            print(f"Error estimating blood pressure: {e}")
            return 120, 80
    
    def calculate_hrv(self, signal_data, fps=30):
        """Calculate Heart Rate Variability"""
        if len(signal_data) < fps * 15:
            return 0
        
        try:
            filtered = signal.medfilt(signal_data, 5)
            peaks, _ = signal.find_peaks(filtered, distance=fps//3)
            
            if len(peaks) < 5:
                return 0
            
            intervals = np.diff(peaks) / fps * 1000
            successive_diffs = np.diff(intervals)
            rmssd = np.sqrt(np.mean(successive_diffs**2))
            
            return min(100, max(10, rmssd))
            
        except Exception as e:
            print(f"Error calculating HRV: {e}")
            return 0
    
    def calculate_stress_index(self, heart_rate, hrv, breathing_rate):
        """Calculate stress index based on multiple parameters"""
        try:
            hr_stress = max(0, (heart_rate - 70) / 50)
            hrv_stress = max(0, (50 - hrv) / 50)
            br_stress = max(0, (breathing_rate - 15) / 15)
            
            stress_index = (hr_stress + hrv_stress + br_stress) / 3
            return min(1.0, max(0.0, stress_index))
            
        except Exception as e:
            print(f"Error calculating stress index: {e}")
            return 0
    
    def calculate_parasympathetic_activity(self, hrv, breathing_rate):
        """Estimate parasympathetic nervous system activity"""
        try:
            hrv_factor = min(1.0, hrv / 50)
            breathing_factor = max(0, (20 - breathing_rate) / 10)
            
            parasympathetic = (hrv_factor + breathing_factor) / 2 * 100
            return min(100, max(0, parasympathetic))
            
        except Exception as e:
            print(f"Error calculating parasympathetic activity: {e}")
            return 50
    
    def calculate_wellness_score(self):
        """Calculate overall wellness score"""
        try:
            hr = self.results['heart_rate']
            hrv = self.results['hrv']
            stress = self.results['stress_index']
            para = self.results['parasympathetic']
            
            hr_score = 1 - abs(hr - 70) / 50 if hr > 0 else 0.5
            hrv_score = min(1, hrv / 50)
            stress_score = 1 - stress
            para_score = para / 100
            
            wellness = (hr_score + hrv_score + stress_score + para_score) / 4 * 100
            return max(0, min(100, wellness))
            
        except Exception as e:
            print(f"Error calculating wellness score: {e}")
            return 50
    
    def update_video(self):
        """Update video feed and process frame"""
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(33, self.update_video)
            return
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            self.face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            self.mp_drawing.draw_landmarks(
                frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            
            if self.monitoring_active:
                ppg_value = self.extract_ppg_signal(frame, face_landmarks.landmark)
                current_time = time.time()
                
                self.ppg_signal.append(ppg_value)
                self.timestamps.append(current_time)
                
                self.session_data['raw_ppg_data'].append(ppg_value)
                self.session_data['timestamps_data'].append(current_time)
                
                # Update progress (30 seconds = 900 frames)
                progress = min(100, (len(self.ppg_signal) / 900) * 100)
                self.progress['value'] = progress
                remaining_time = max(0, 30 - len(self.ppg_signal)/30)
                self.progress_label.config(text=f"Recording: {len(self.ppg_signal)}/900 frames - {remaining_time:.1f}s remaining")
                
                # Auto-stop after 30 seconds
                if len(self.ppg_signal) >= 900:
                    self.stop_monitoring()
                
                # Start calculating after 10 seconds (300 frames)
                elif len(self.ppg_signal) >= 300:
                    self.calculate_all_metrics(face_landmarks.landmark)
        else:
            self.face_detected = False
            
        # Convert frame for display
        frame_pil = Image.fromarray(frame)
        frame_pil = frame_pil.resize((500, 375))
        frame_tk = ImageTk.PhotoImage(frame_pil)
        
        self.video_label.configure(image=frame_tk)
        self.video_label.image = frame_tk
        
        status = "Face Detected" if self.face_detected else "No Face Detected"
        if self.monitoring_active:
            status += f" - Recording ({len(self.ppg_signal)}/900)"
        
        self.root.title(f"Face Vital - Health Monitor - {status}")
        
        self.root.after(33, self.update_video)
    
    def calculate_all_metrics(self, landmarks):
        """Calculate all health metrics with faster updates"""
        if len(self.ppg_signal) < 300:  # Start after 10 seconds
            return
        
        signal_array = np.array(list(self.ppg_signal))
        
        # Calculate heart rate
        hr = self.calculate_heart_rate(signal_array, list(self.timestamps))
        self.results['heart_rate'] = int(hr) if hr > 0 else 0
        self.hr_values.append(self.results['heart_rate'])
        
        # Calculate breathing rate
        br = self.calculate_breathing_rate(signal_array)
        self.results['breathing_rate'] = int(br) if br > 0 else 0
        self.br_values.append(self.results['breathing_rate'])
        
        # Calculate HRV
        hrv = self.calculate_hrv(signal_array)
        self.results['hrv'] = int(hrv)
        self.hrv_values.append(self.results['hrv'])
        
        # Calculate stress index
        stress = self.calculate_stress_index(hr, hrv, br)
        self.results['stress_index'] = round(stress, 2)
        self.stress_values.append(self.results['stress_index'])
        
        # Calculate parasympathetic activity
        para = self.calculate_parasympathetic_activity(hrv, br)
        self.results['parasympathetic'] = int(para)
        self.para_values.append(self.results['parasympathetic'])
        
        # Estimate blood pressure
        sys_bp, dia_bp = self.estimate_blood_pressure(hr, hrv, stress)
        self.results['blood_pressure_sys'] = sys_bp
        self.results['blood_pressure_dia'] = dia_bp
        self.bp_sys_values.append(sys_bp)
        self.bp_dia_values.append(dia_bp)
        
        # Calculate wellness score
        wellness = self.calculate_wellness_score()
        self.results['wellness_score'] = int(wellness)
        self.wellness_values.append(self.results['wellness_score'])
        
        # Increment calculation count
        self.calculation_count += 1
        
        # Store measurement for export
        measurement = {
            'timestamp': datetime.now().isoformat(),
            'heart_rate': self.results['heart_rate'],
            'breathing_rate': self.results['breathing_rate'],
            'blood_pressure_sys': self.results['blood_pressure_sys'],
            'blood_pressure_dia': self.results['blood_pressure_dia'],
            'hrv': self.results['hrv'],
            'stress_index': self.results['stress_index'],
            'parasympathetic': self.results['parasympathetic'],
            'wellness_score': self.results['wellness_score']
        }
        self.session_data['measurements'].append(measurement)
        
        # Update UI
        self.update_results_display()
        
        # Update metric plots (called manually, not via animation)
        self.update_metrics_plots()
    
    def update_results_display(self):
        """Update the results display with color coding"""
        # Heart rate
        hr = self.results['heart_rate']
        hr_color = 'green' if 60 <= hr <= 100 else 'orange' if 50 <= hr <= 120 else 'red'
        self.result_labels['heart_rate'].config(text=f"{hr} bpm", foreground=hr_color)
        
        # Breathing rate
        br = self.results['breathing_rate']
        br_color = 'green' if 12 <= br <= 20 else 'orange' if 8 <= br <= 25 else 'red'
        self.result_labels['breathing_rate'].config(text=f"{br} rpm", foreground=br_color)
        
        # Blood pressure
        sys_bp = self.results['blood_pressure_sys']
        dia_bp = self.results['blood_pressure_dia']
        bp_color = 'green' if sys_bp < 130 and dia_bp < 85 else 'orange' if sys_bp < 140 and dia_bp < 90 else 'red'
        self.result_labels['blood_pressure'].config(
            text=f"{sys_bp}/{dia_bp} mmHg", foreground=bp_color)
        
        # HRV
        hrv = self.results['hrv']
        hrv_color = 'green' if hrv > 40 else 'orange' if hrv > 20 else 'red'
        self.result_labels['hrv'].config(text=f"{hrv} ms", foreground=hrv_color)
        
        # Stress index
        stress = self.results['stress_index']
        stress_color = 'green' if stress < 0.3 else 'orange' if stress < 0.7 else 'red'
        self.result_labels['stress_index'].config(text=f"{stress}", foreground=stress_color)
        
        # Parasympathetic
        para = self.results['parasympathetic']
        para_color = 'green' if para > 60 else 'orange' if para > 30 else 'red'
        self.result_labels['parasympathetic'].config(text=f"{para} %", foreground=para_color)
        
        # Wellness score
        wellness = self.results['wellness_score']
        wellness_color = 'green' if wellness > 70 else 'orange' if wellness > 40 else 'red'
        self.result_labels['wellness_score'].config(text=f"{wellness} /100", foreground=wellness_color)
    
    def start_monitoring(self):
        """Start health monitoring"""
        if not self.face_detected:
            messagebox.showwarning("Warning", "Please ensure your face is visible in the camera!")
            return
    
        self.monitoring_active = True
        self.calculation_count = 0
    
    # Clear all data
        self.ppg_signal.clear()
        self.timestamps.clear()
        self.hr_values.clear()
        self.br_values.clear()
        self.hrv_values.clear()
        self.stress_values.clear()
        self.para_values.clear()
        self.wellness_values.clear()
        self.bp_sys_values.clear()
        self.bp_dia_values.clear()
    
        self.progress['value'] = 0
    
    # Initialize session data
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'measurements': [],
            'raw_ppg_data': [],
            'timestamps_data': []
        }
    
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.DISABLED)
        if PDF_AVAILABLE and hasattr(self, 'pdf_btn'):
            self.pdf_btn.config(state=tk.DISABLED)
    
        self.progress_label.config(text="Starting 30-second monitoring...")
        messagebox.showinfo("Started", "30-second health monitoring started! Please remain still and look at the camera.")

    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        self.session_data['end_time'] = datetime.now().isoformat()
    
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.NORMAL)
    
    # Enable PDF button after monitoring stops
        if PDF_AVAILABLE and hasattr(self, 'pdf_btn'):
            self.pdf_btn.config(state=tk.NORMAL)
    
        self.progress['value'] = 100
        self.progress_label.config(text="Monitoring completed! Data ready for export.")
    
        messagebox.showinfo(
            "Completed",
            f"30-second monitoring completed! {self.calculation_count} calculations performed. "
            f"You can now export data or generate PDF report."
        )


    def save_plot_as_image(self, fig, filename):
        """Save a matplotlib figure as an image and return the image data"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()
    
    def generate_pdf_report(self):
        """Generate a comprehensive PDF report with wave images"""
        if not PDF_AVAILABLE:
            messagebox.showerror(
                "PDF Library Missing", 
                "PDF generation requires the 'reportlab' library.\n\n"
                "Please install it using:\n"
                "pip install reportlab\n\n"
                "Then restart the application."
            )
            return
        
        if not self.session_data['measurements']:
            messagebox.showwarning("No Data", "No monitoring data available to generate report!\n\nPlease complete a monitoring session first.")
            return
    
    # Ask user for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save Health Monitoring PDF Report",
            initialfilename=f"Health_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    
        if not filename:
            return
    
        try:
        # Show progress
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Generating PDF Report...")
            progress_window.geometry("400x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
        
            progress_label = ttk.Label(progress_window, text="Generating PDF report, please wait...")
            progress_label.pack(pady=20)
        
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10, padx=20, fill=tk.X)
            progress_bar.start()
        
            progress_window.update()
        
        # Create PDF document
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
        
        # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("30-SECOND HEALTH MONITORING REPORT", title_style))
            story.append(Spacer(1, 20))
        
        # Session Information
            story.append(Paragraph("SESSION INFORMATION", styles['Heading2']))
            session_data = [
                ['Parameter', 'Value'],
                ['Start Time', self.session_data['start_time'][:19].replace('T', ' ')],  # Format datetime
                ['End Time', self.session_data['end_time'][:19].replace('T', ' ') if self.session_data['end_time'] else 'N/A'],
                ['Duration', '30 seconds'],
                ['Total Frames Captured', str(len(self.session_data['raw_ppg_data']))],
                ['Total Measurements', str(len(self.session_data['measurements']))],
                ['Calculations Performed', str(self.calculation_count)]
            ]
        
            session_table = Table(session_data, colWidths=[3*inch, 3*inch])
            session_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(session_table)
            story.append(Spacer(1, 20))
        
        # Final Measurements
            story.append(Paragraph("FINAL HEALTH MEASUREMENTS", styles['Heading2']))
            measurements_data = [
                ['Metric', 'Value', 'Status'],
                ['Heart Rate', f"{self.results['heart_rate']} bpm", self.get_status_text('heart_rate')],
                ['Breathing Rate', f"{self.results['breathing_rate']} rpm", self.get_status_text('breathing_rate')],
                ['Blood Pressure', f"{self.results['blood_pressure_sys']}/{self.results['blood_pressure_dia']} mmHg", self.get_status_text('blood_pressure')],
                ['HRV', f"{self.results['hrv']} ms", self.get_status_text('hrv')],
                ['Stress Index', f"{self.results['stress_index']}", self.get_status_text('stress_index')],
                ['Parasympathetic Activity', f"{self.results['parasympathetic']}%", self.get_status_text('parasympathetic')],
                ['Wellness Score', f"{self.results['wellness_score']}/100", self.get_status_text('wellness_score')]
            ]
        
            measurements_table = Table(measurements_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
            measurements_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(measurements_table)
            story.append(PageBreak())
        
        # Add wave plots
            story.append(Paragraph("HEALTH METRICS TREND ANALYSIS", styles['Heading2']))
            story.append(Spacer(1, 10))
        
        # Create individual metric plots for PDF
            metrics_to_plot = [
                ("Heart Rate", "heart_rate", self.hr_values, "bpm", "red"),
                ("Breathing Rate", "breathing_rate", self.br_values, "rpm", "blue"),
                ("HRV", "hrv", self.hrv_values, "ms", "green"),
                ("Stress Index", "stress_index", self.stress_values, "", "orange"),
                ("Parasympathetic Activity", "parasympathetic", self.para_values, "%", "purple"),
                ("Wellness Score", "wellness_score", self.wellness_values, "/100", "teal")
            ]
        
            for name, key, data_deque, unit, color in metrics_to_plot:
                if len(data_deque) > 1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    time_axis = np.linspace(-len(data_deque), 0, len(data_deque))
                    ax.plot(time_axis, list(data_deque), color=color, linewidth=2)
                    ax.set_title(f'{name} Trend Over Time', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Time (relative seconds)')
                    ax.set_ylabel(f'{name} ({unit})')
                    ax.grid(True, alpha=0.3)
                
                # Save plot as image
                    img_data = self.save_plot_as_image(fig, f'{key}_plot.png')
                    img = RLImage(io.BytesIO(img_data), width=6*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                    plt.close(fig)
        
        # Blood pressure plot
            if len(self.bp_sys_values) > 1 and len(self.bp_dia_values) > 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                time_axis = np.linspace(-len(self.bp_sys_values), 0, len(self.bp_sys_values))
                ax.plot(time_axis, list(self.bp_sys_values), 'darkred', linewidth=2, label='Systolic')
                ax.plot(time_axis, list(self.bp_dia_values), 'maroon', linewidth=2, label='Diastolic')
                ax.set_title('Blood Pressure Trend Over Time', fontsize=14, fontweight='bold')
                ax.set_xlabel('Time (relative seconds)')
                ax.set_ylabel('Blood Pressure (mmHg)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
                img_data = self.save_plot_as_image(fig, 'bp_plot.png')
                img = RLImage(io.BytesIO(img_data), width=6*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 10))
                plt.close(fig)
        
            story.append(PageBreak())
        
        # Raw signal plot
            story.append(Paragraph("RAW SIGNAL ANALYSIS", styles['Heading2']))
        
            if len(self.ppg_signal) > 10:
                fig, ax = plt.subplots(figsize=(10, 4))
                time_axis = np.linspace(-len(self.ppg_signal)/30, 0, len(self.ppg_signal))
                ax.plot(time_axis, list(self.ppg_signal), 'b-', linewidth=1)
                ax.set_title('Raw PPG Signal', fontsize=14, fontweight='bold')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
            
                img_data = self.save_plot_as_image(fig, 'raw_ppg.png')
                img = RLImage(io.BytesIO(img_data), width=7*inch, height=3*inch)
                story.append(img)
                plt.close(fig)
        
        # Health interpretation and recommendations
            story.append(PageBreak())
            story.append(Paragraph("HEALTH INTERPRETATION", styles['Heading2']))
            interpretation = self.get_health_interpretation()
            story.append(Paragraph(interpretation, styles['Normal']))
            story.append(Spacer(1, 20))
        
            story.append(Paragraph("RECOMMENDATIONS", styles['Heading2']))
            recommendations = self.get_recommendations()
            story.append(Paragraph(recommendations, styles['Normal']))
        
        # Build PDF
            doc.build(story)
        
        # Close progress window
            progress_bar.stop()
            progress_window.destroy()
        
            messagebox.showinfo("Success", f"PDF report generated successfully!\n\nSaved as: {filename}")
        
        except Exception as e:
        # Close progress window if it exists
            try:
                progress_bar.stop()
                progress_window.destroy()
            except:
                pass
        
            error_msg = f"Failed to generate PDF report:\n\n{str(e)}\n\nPlease ensure you have sufficient disk space and write permissions."
            messagebox.showerror("PDF Generation Error", error_msg)
            print(f"PDF generation error: {e}")
            import traceback
            traceback.print_exc()
    def get_status_text(self, metric):
        """Get status text for a metric"""
        if metric == 'heart_rate':
            hr = self.results['heart_rate']
            if 60 <= hr <= 100:
                return "Normal"
            elif 50 <= hr <= 120:
                return "Acceptable"
            else:
                return "Abnormal"
        elif metric == 'breathing_rate':
            br = self.results['breathing_rate']
            if 12 <= br <= 20:
                return "Normal"
            elif 8 <= br <= 25:
                return "Acceptable"
            else:
                return "Abnormal"
        elif metric == 'blood_pressure':
            sys_bp = self.results['blood_pressure_sys']
            dia_bp = self.results['blood_pressure_dia']
            if sys_bp < 130 and dia_bp < 85:
                return "Normal"
            elif sys_bp < 140 and dia_bp < 90:
                return "Elevated"
            else:
                return "High"
        elif metric == 'hrv':
            hrv = self.results['hrv']
            if hrv > 40:
                return "Good"
            elif hrv > 20:
                return "Fair"
            else:
                return "Poor"
        elif metric == 'stress_index':
            stress = self.results['stress_index']
            if stress < 0.3:
                return "Low"
            elif stress < 0.7:
                return "Moderate"
            else:
                return "High"
        elif metric == 'parasympathetic':
            para = self.results['parasympathetic']
            if para > 60:
                return "Good"
            elif para > 30:
                return "Fair"
            else:
                return "Poor"
        elif metric == 'wellness_score':
            wellness = self.results['wellness_score']
            if wellness > 70:
                return "Excellent"
            elif wellness > 40:
                return "Good"
            else:
                return "Needs Improvement"
        return "Unknown"
    
    def export_data(self):
        """Export session data to files"""
        if not self.session_data['measurements']:
            messagebox.showwarning("Warning", "No data to export!")
            return
        
        # Ask user for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ],
            title="Save Health Monitoring Data"
        )
        
        if not filename:
            return
        
        try:
            # Get file extension
            file_ext = filename.lower().split('.')[-1]
            
            if file_ext == 'csv':
                self.export_to_csv(filename)
            elif file_ext == 'json':
                self.export_to_json(filename)
            elif file_ext == 'xlsx':
                self.export_to_excel(filename)
            else:
                # Default to CSV
                self.export_to_csv(filename + '.csv')
            
            messagebox.showinfo("Success", f"Data exported successfully to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def export_to_csv(self, filename):
        """Export data to CSV format"""
        # Create summary data
        summary_data = {
            'Session Start': [self.session_data['start_time']],
            'Session End': [self.session_data['end_time']],
            'Duration (seconds)': [30],
            'Total Measurements': [len(self.session_data['measurements'])],
            'Calculations Performed': [self.calculation_count]
        }
        
        # Add final measurements
        if self.session_data['measurements']:
            final_measurement = self.session_data['measurements'][-1]
            for key, value in final_measurement.items():
                if key != 'timestamp':
                    summary_data[key.replace('_', ' ').title()] = [value]
        
        # Create DataFrames
        summary_df = pd.DataFrame(summary_data)
        measurements_df = pd.DataFrame(self.session_data['measurements'])
        
        # Write to CSV with multiple sections
        with open(filename, 'w', newline='') as f:
            f.write("=== 30-SECOND HEALTH MONITORING SESSION SUMMARY ===\n")
            summary_df.to_csv(f, index=False)
            f.write("\n=== DETAILED MEASUREMENTS TIMELINE ===\n")
            measurements_df.to_csv(f, index=False)
            
            # Add raw PPG data if available
            if self.session_data['raw_ppg_data']:
                f.write("\n=== RAW PPG SIGNAL DATA (900 FRAMES) ===\n")
                ppg_df = pd.DataFrame({
                    'frame_number': range(len(self.session_data['raw_ppg_data'])),
                    'timestamp': self.session_data['timestamps_data'][:len(self.session_data['raw_ppg_data'])],
                    'ppg_value': self.session_data['raw_ppg_data']
                })
                ppg_df.to_csv(f, index=False)
    
    def export_to_json(self, filename):
        """Export data to JSON format"""
        export_data = {
            'session_info': {
                'start_time': self.session_data['start_time'],
                'end_time': self.session_data['end_time'],
                'duration_seconds': 30,
                'total_frames': len(self.session_data['raw_ppg_data']),
                'total_measurements': len(self.session_data['measurements']),
                'calculations_performed': self.calculation_count
            },
            'final_results': self.results,
            'measurements_timeline': self.session_data['measurements'],
            'raw_data': {
                'ppg_signal': self.session_data['raw_ppg_data'],
                'timestamps': self.session_data['timestamps_data'][:len(self.session_data['raw_ppg_data'])],
                'heart_rate_history': list(self.hr_values),
                'breathing_rate_history': list(self.br_values),
                'hrv_history': list(self.hrv_values),
                'stress_history': list(self.stress_values),
                'parasympathetic_history': list(self.para_values),
                'wellness_history': list(self.wellness_values),
                'bp_systolic_history': list(self.bp_sys_values),
                'bp_diastolic_history': list(self.bp_dia_values)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_to_excel(self, filename):
        """Export data to Excel format with multiple sheets"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': ['Session Start', 'Session End', 'Duration (seconds)', 'Total Frames', 'Total Measurements', 'Calculations Performed'],
                    'Value': [
                        self.session_data['start_time'],
                        self.session_data['end_time'],
                        30,
                        len(self.session_data['raw_ppg_data']),
                        len(self.session_data['measurements']),
                        self.calculation_count
                    ]
                }
                
                # Add final results
                for key, value in self.results.items():
                    summary_data['Metric'].append(key.replace('_', ' ').title())
                    summary_data['Value'].append(value)
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Measurements timeline
                if self.session_data['measurements']:
                    measurements_df = pd.DataFrame(self.session_data['measurements'])
                    measurements_df.to_excel(writer, sheet_name='Measurements Timeline', index=False)
                
                # Wave data histories
                wave_data = {
                    'Time_Point': range(len(self.hr_values)),
                    'Heart_Rate': list(self.hr_values),
                    'Breathing_Rate': list(self.br_values),
                    'HRV': list(self.hrv_values),
                    'Stress_Index': list(self.stress_values),
                    'Parasympathetic': list(self.para_values),
                    'Wellness_Score': list(self.wellness_values),
                    'Systolic_BP': list(self.bp_sys_values),
                    'Diastolic_BP': list(self.bp_dia_values)
                }
                wave_df = pd.DataFrame(wave_data)
                wave_df.to_excel(writer, sheet_name='Wave Data', index=False)
                
                # Raw PPG data
                if self.session_data['raw_ppg_data']:
                    ppg_df = pd.DataFrame({
                        'Frame_Number': range(len(self.session_data['raw_ppg_data'])),
                        'Timestamp': self.session_data['timestamps_data'][:len(self.session_data['raw_ppg_data'])],
                        'PPG_Value': self.session_data['raw_ppg_data']
                    })
                    ppg_df.to_excel(writer, sheet_name='Raw PPG Signal', index=False)
                    
        except ImportError:
            # Fallback to CSV if openpyxl is not available
            messagebox.showwarning("Warning", "Excel export requires openpyxl. Saving as CSV instead.")
            self.export_to_csv(filename.replace('.xlsx', '.csv'))
    
    def get_health_interpretation(self):
        """Get health interpretation based on measurements"""
        interpretations = []
        
        # Heart Rate
        hr = self.results['heart_rate']
        if hr < 60:
            interpretations.append("• Heart Rate: Below normal (Bradycardia) - Consider consulting a healthcare provider.")
        elif hr > 100:
            interpretations.append("• Heart Rate: Above normal (Tachycardia) - May indicate stress, exercise, or other factors.")
        else:
            interpretations.append("• Heart Rate: Normal range (60-100 bpm).")
        
        # Blood Pressure
        sys_bp = self.results['blood_pressure_sys']
        dia_bp = self.results['blood_pressure_dia']
        if sys_bp < 120 and dia_bp < 80:
            interpretations.append("• Blood Pressure: Normal (<120/80 mmHg).")
        elif sys_bp < 130 and dia_bp < 85:
            interpretations.append("• Blood Pressure: Normal to slightly elevated.")
        elif sys_bp < 140 and dia_bp < 90:
            interpretations.append("• Blood Pressure: Stage 1 hypertension - monitor regularly.")
        else:
            interpretations.append("• Blood Pressure: High (≥140/90 mmHg) - Recommend medical evaluation.")
        
        # HRV
        hrv = self.results['hrv']
        if hrv > 40:
            interpretations.append("• Heart Rate Variability: Good - indicates healthy autonomic nervous system.")
        elif hrv > 20:
            interpretations.append("• Heart Rate Variability: Fair - room for improvement through stress management.")
        else:
            interpretations.append("• Heart Rate Variability: Low - may indicate stress or autonomic dysfunction.")
        
        # Stress Index
        stress = self.results['stress_index']
        if stress < 0.3:
            interpretations.append("• Stress Level: Low - maintaining good stress management.")
        elif stress < 0.7:
            interpretations.append("• Stress Level: Moderate - consider stress reduction techniques.")
        else:
            interpretations.append("• Stress Level: High - recommend stress management and relaxation practices.")
        
        # Wellness Score
        wellness = self.results['wellness_score']
        if wellness > 70:
            interpretations.append("• Overall Wellness: Excellent - maintaining good health habits.")
        elif wellness > 40:
            interpretations.append("• Overall Wellness: Good - some areas for improvement identified.")
        else:
            interpretations.append("• Overall Wellness: Needs attention - consider comprehensive health evaluation.")
        
        return "\n".join(interpretations)
    
    def get_recommendations(self):
        """Get health recommendations based on measurements"""
        recommendations = []
        
        hr = self.results['heart_rate']
        stress = self.results['stress_index']
        hrv = self.results['hrv']
        wellness = self.results['wellness_score']
        
        # General recommendations
        recommendations.append("GENERAL RECOMMENDATIONS:")
        recommendations.append("• Maintain regular exercise routine (150 minutes moderate activity per week)")
        recommendations.append("• Practice stress management techniques (meditation, deep breathing)")
        recommendations.append("• Ensure adequate sleep (7-9 hours per night)")
        recommendations.append("• Stay hydrated and maintain balanced nutrition")
        
        # Specific recommendations based on results
        if stress > 0.5:
            recommendations.append("\nSTRESS MANAGEMENT:")
            recommendations.append("• Consider mindfulness meditation or yoga")
            recommendations.append("• Practice progressive muscle relaxation")
            recommendations.append("• Limit caffeine intake")
            recommendations.append("• Ensure work-life balance")
        
        if hrv < 30:
            recommendations.append("\nHEART RATE VARIABILITY IMPROVEMENT:")
            recommendations.append("• Regular cardiovascular exercise")
            recommendations.append("• Breathing exercises (4-7-8 technique)")
            recommendations.append("• Reduce alcohol consumption")
            recommendations.append("• Consider heart rate variability training")
        
        if hr > 100 or hr < 60:
            recommendations.append("\nHEART RATE CONCERNS:")
            recommendations.append("• Monitor heart rate regularly")
            recommendations.append("• Consult healthcare provider if persistent")
            recommendations.append("• Avoid excessive caffeine")
            recommendations.append("• Maintain regular sleep schedule")
        
        if wellness < 50:
            recommendations.append("\nWELLNESS IMPROVEMENT:")
            recommendations.append("• Comprehensive health assessment recommended")
            recommendations.append("• Consider lifestyle modifications")
            recommendations.append("• Regular monitoring of vital signs")
            recommendations.append("• Professional health consultation advised")
        
        recommendations.append("\nDISCLAIMER:")
        recommendations.append("This is a demonstration tool for educational purposes. ")
        recommendations.append("Measurements are estimates and should not replace professional medical advice. ")
        recommendations.append("Consult healthcare professionals for medical concerns.")
        
        return "\n".join(recommendations)
    
    def __del__(self):
        """Cleanup when application closes"""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
        except:
            pass

def main():
    """Main application entry point"""
    try:
        app = FaceVitalMonitor()
        app.root.protocol("WM_DELETE_WINDOW", lambda: (app.cap.release(), app.root.destroy()))
        app.root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()