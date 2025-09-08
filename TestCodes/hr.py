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

class FaceVitalMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Vital - Health Monitor (MediaPipe)")
        self.root.geometry("1400x900")
        
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
        
        # Data storage
        self.ppg_signal = deque(maxlen=900)  # 30 seconds at 30fps
        self.timestamps = deque(maxlen=900)
        self.hr_signal = deque(maxlen=300)   # For HR visualization
        self.rr_signal = deque(maxlen=300)   # For RR visualization
        self.face_detected = False
        self.monitoring_active = False
        
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
        
        self.start_btn = ttk.Button(button_frame, text="Start Monitoring", 
                                   command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Monitoring", 
                                  command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = ttk.Button(button_frame, text="Export Data", 
                                    command=self.export_data, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(video_frame, length=400, mode='determinate')
        self.progress.pack(pady=10)
        
        # Progress label
        self.progress_label = ttk.Label(video_frame, text="Ready to start monitoring")
        self.progress_label.pack(pady=5)
        
        # Right side - Results and graphs
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Health Metrics")
        
        # Pulse waves tab
        waves_frame = ttk.Frame(notebook)
        notebook.add(waves_frame, text="Pulse Waves")
        
        self.setup_results_tab(results_frame)
        self.setup_waves_tab(waves_frame)
        
        # Start video feed
        self.update_video()
    
    def setup_results_tab(self, parent):
        """Setup the results display tab"""
        # Results display
        results_display = ttk.LabelFrame(parent, text="Current Measurements", padding="10")
        results_display.pack(fill=tk.BOTH, expand=True)
        
        # Create result labels
        self.result_labels = {}
        metrics = [
            ("Heart Rate", "heart_rate", "bpm"),
            ("Breathing Rate", "breathing_rate", "rpm"),
            ("Blood Pressure", "blood_pressure", "mmHg"),
            ("HRV", "hrv", "ms"),
            ("Stress Index", "stress_index", ""),
            ("Parasympathetic", "parasympathetic", "%"),
            ("Wellness Score", "wellness_score", "/100")
        ]
        
        for i, (name, key, unit) in enumerate(metrics):
            frame = ttk.Frame(results_display)
            frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(frame, text=f"{name}:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
            
            if key == "blood_pressure":
                self.result_labels[key] = ttk.Label(frame, text="--/-- mmHg", 
                                                   font=('Arial', 14), foreground='blue')
            else:
                self.result_labels[key] = ttk.Label(frame, text=f"-- {unit}", 
                                                   font=('Arial', 14), foreground='blue')
            self.result_labels[key].pack(side=tk.RIGHT)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(parent, text="Instructions", padding="10")
        instructions_frame.pack(fill=tk.X, pady=10)
        
        instructions_text = """
1. Sit comfortably in front of the camera
2. Ensure your face is well-lit and clearly visible
3. Look directly at the camera and minimize movement
4. Click "Start Monitoring" and wait for 30 seconds
5. Keep still during the entire measurement period
6. Click "Stop Monitoring" when finished
7. Use "Export Data" to save your measurements
        """
        
        ttk.Label(instructions_frame, text=instructions_text, 
                 font=('Arial', 10), justify=tk.LEFT).pack(anchor=tk.W)
    
    def setup_waves_tab(self, parent):
        """Setup the pulse waves visualization tab"""
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 8))
        self.fig.tight_layout(pad=3.0)
        
        # PPG Signal plot
        self.ax1.set_title('Raw PPG Signal', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=1)
        
        # Heart Rate Signal plot
        self.ax2.set_title('Heart Rate Variability', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('HR (bpm)')
        self.ax2.grid(True, alpha=0.3)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2)
        
        # Breathing Rate Signal plot
        self.ax3.set_title('Breathing Pattern', fontsize=12, fontweight='bold')
        self.ax3.set_ylabel('Amplitude')
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.grid(True, alpha=0.3)
        self.line3, = self.ax3.plot([], [], 'g-', linewidth=2)
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start animation
        self.anim = FuncAnimation(self.fig, self.update_plots, interval=100, blit=False)
    
    def update_plots(self, frame):
        """Update the pulse wave plots"""
        if len(self.ppg_signal) < 10:
            return self.line1, self.line2, self.line3
        
        # Time axis (last 30 seconds)
        time_axis = np.linspace(-len(self.ppg_signal)/30, 0, len(self.ppg_signal))
        
        # Update PPG signal plot
        self.line1.set_data(time_axis, list(self.ppg_signal))
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update HR signal plot
        if len(self.hr_signal) > 0:
            hr_time = np.linspace(-len(self.hr_signal), 0, len(self.hr_signal))
            self.line2.set_data(hr_time, list(self.hr_signal))
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        # Update breathing signal plot (filtered PPG for breathing)
        if len(self.ppg_signal) > 60:
            try:
                # Filter for breathing rate
                signal_array = np.array(list(self.ppg_signal))
                nyquist = 15  # 30fps / 2
                low = 0.1 / nyquist
                high = 0.5 / nyquist
                b, a = signal.butter(2, [low, high], btype='band')
                breathing_signal = signal.filtfilt(b, a, signal_array)
                
                self.line3.set_data(time_axis, breathing_signal)
                self.ax3.relim()
                self.ax3.autoscale_view()
            except:
                pass
        
        return self.line1, self.line2, self.line3
    
    def extract_ppg_signal(self, frame, landmarks):
        """Extract PPG signal from facial regions using MediaPipe landmarks"""
        try:
            h, w = frame.shape[:2]
            
            # Define ROI indices for forehead and cheek regions (MediaPipe face mesh indices)
            forehead_indices = [10, 151, 9, 10, 151, 9, 10, 151]  # Forehead region
            left_cheek_indices = [116, 117, 118, 119, 120, 121]   # Left cheek
            right_cheek_indices = [345, 346, 347, 348, 349, 350]  # Right cheek
            
            roi_values = []
            
            for indices in [forehead_indices, left_cheek_indices, right_cheek_indices]:
                # Get region coordinates
                region_points = []
                for idx in indices:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * w)
                        y = int(landmarks[idx].y * h)
                        region_points.append([x, y])
                
                if len(region_points) > 2:
                    # Create mask for ROI
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(region_points)], 255)
                    
                    # Extract green channel (most sensitive to blood volume changes)
                    green_channel = frame[:, :, 1]
                    roi_mean = cv2.mean(green_channel, mask)[0]
                    roi_values.append(roi_mean)
            
            # Average the ROI values
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
        if len(signal_data) < fps * 10:  # Need at least 10 seconds
            return 0
        
        try:
            # Detrend signal
            detrended = signal.detrend(signal_data)
            
            # Apply bandpass filter (0.5-4 Hz for heart rate)
            nyquist = fps / 2
            low = 0.5 / nyquist
            high = 4.0 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, detrended)
            
            # FFT analysis
            fft_data = fft(filtered)
            freqs = np.fft.fftfreq(len(filtered), 1/fps)
            
            # Find peak in heart rate range
            valid_indices = (freqs >= 0.5) & (freqs <= 4.0)
            valid_fft = np.abs(fft_data[valid_indices])
            valid_freqs = freqs[valid_indices]
            
            if len(valid_fft) > 0:
                peak_idx = np.argmax(valid_fft)
                heart_rate_hz = valid_freqs[peak_idx]
                heart_rate_bpm = heart_rate_hz * 60
                return max(50, min(200, heart_rate_bpm))  # Clamp to reasonable range
            
        except Exception as e:
            print(f"Error calculating heart rate: {e}")
        
        return 0
    
    def calculate_breathing_rate(self, signal_data, fps=30):
        """Calculate breathing rate from signal variations"""
        if len(signal_data) < fps * 15:
            return 0
        
        try:
            # Low-pass filter for breathing rate (0.1-0.5 Hz)
            nyquist = fps / 2
            low = 0.1 / nyquist
            high = 0.5 / nyquist
            b, a = signal.butter(2, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, signal_data)
            
            # Count peaks
            peaks, _ = signal.find_peaks(filtered, distance=fps*2)  # Minimum 2 seconds between breaths
            breathing_rate = len(peaks) * (60 / (len(signal_data) / fps))
            
            return max(5, min(30, breathing_rate))  # Clamp to reasonable range
            
        except Exception as e:
            print(f"Error calculating breathing rate: {e}")
            return 0
    
    def estimate_blood_pressure(self, heart_rate, hrv, stress_index):
        """Estimate blood pressure using correlation with HR, HRV, and stress"""
        try:
            # Empirical estimation based on research correlations
            base_sys = 120
            base_dia = 80
            
            # Adjust based on heart rate
            hr_factor = (heart_rate - 70) * 0.5
            
            # Adjust based on stress
            stress_factor = stress_index * 10
            
            # Adjust based on HRV (lower HRV = higher BP)
            hrv_factor = (50 - hrv) * 0.2
            
            sys_bp = base_sys + hr_factor + stress_factor + hrv_factor
            dia_bp = base_dia + hr_factor * 0.6 + stress_factor * 0.6 + hrv_factor * 0.6
            
            # Clamp to reasonable ranges
            sys_bp = max(90, min(180, sys_bp))
            dia_bp = max(60, min(120, dia_bp))
            
            return int(sys_bp), int(dia_bp)
            
        except Exception as e:
            print(f"Error estimating blood pressure: {e}")
            return 120, 80
    
    def calculate_hrv(self, signal_data, fps=30):
        """Calculate Heart Rate Variability"""
        if len(signal_data) < fps * 20:
            return 0
        
        try:
            # Find R-R intervals (simplified)
            filtered = signal.medfilt(signal_data, 5)
            peaks, _ = signal.find_peaks(filtered, distance=fps//3)
            
            if len(peaks) < 5:
                return 0
            
            # Calculate intervals
            intervals = np.diff(peaks) / fps * 1000  # Convert to milliseconds
            
            # RMSSD (Root Mean Square of Successive Differences)
            successive_diffs = np.diff(intervals)
            rmssd = np.sqrt(np.mean(successive_diffs**2))
            
            return min(100, max(10, rmssd))
            
        except Exception as e:
            print(f"Error calculating HRV: {e}")
            return 0
    
    def calculate_stress_index(self, heart_rate, hrv, breathing_rate):
        """Calculate stress index based on multiple parameters"""
        try:
            # Normalized stress factors
            hr_stress = max(0, (heart_rate - 70) / 50)  # Normal resting HR ~70
            hrv_stress = max(0, (50 - hrv) / 50)        # Lower HRV = higher stress
            br_stress = max(0, (breathing_rate - 15) / 15)  # Normal breathing ~15
            
            stress_index = (hr_stress + hrv_stress + br_stress) / 3
            return min(1.0, max(0.0, stress_index))
            
        except Exception as e:
            print(f"Error calculating stress index: {e}")
            return 0
    
    def calculate_parasympathetic_activity(self, hrv, breathing_rate):
        """Estimate parasympathetic nervous system activity"""
        try:
            # Higher HRV and slower breathing indicate higher parasympathetic activity
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
            
            # Scoring factors (0-1 scale)
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
        
        frame = cv2.flip(frame, 1)  # Mirror image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            self.face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw face landmarks
            self.mp_drawing.draw_landmarks(
                frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            
            # Extract PPG signal if monitoring
            if self.monitoring_active:
                ppg_value = self.extract_ppg_signal(frame, face_landmarks.landmark)
                current_time = time.time()
                
                self.ppg_signal.append(ppg_value)
                self.timestamps.append(current_time)
                
                # Store data for export
                self.session_data['raw_ppg_data'].append(ppg_value)
                self.session_data['timestamps_data'].append(current_time)
                
                # Update progress
                progress = min(100, (len(self.ppg_signal) / 900) * 100)
                self.progress['value'] = progress
                self.progress_label.config(text=f"Recording: {len(self.ppg_signal)}/900 frames ({progress:.1f}%)")
                
                # Calculate metrics if we have enough data
                if len(self.ppg_signal) >= 300:  # 10 seconds minimum
                    self.calculate_all_metrics(face_landmarks.landmark)
        else:
            self.face_detected = False
            
        # Convert frame for display
        frame_pil = Image.fromarray(frame)
        frame_pil = frame_pil.resize((500, 375))
        frame_tk = ImageTk.PhotoImage(frame_pil)
        
        self.video_label.configure(image=frame_tk)
        self.video_label.image = frame_tk
        
        # Add status text
        status = "Face Detected" if self.face_detected else "No Face Detected"
        if self.monitoring_active:
            status += f" - Recording"
        
        self.root.title(f"Face Vital - Health Monitor - {status}")
        
        self.root.after(33, self.update_video)  # ~30 FPS
    
    def calculate_all_metrics(self, landmarks):
        """Calculate all health metrics"""
        if len(self.ppg_signal) < 300:
            return
        
        signal_array = np.array(list(self.ppg_signal))
        
        # Calculate heart rate
        hr = self.calculate_heart_rate(signal_array, list(self.timestamps))
        self.results['heart_rate'] = int(hr) if hr > 0 else 0
        
        # Add to HR signal for visualization
        self.hr_signal.append(self.results['heart_rate'])
        
        # Calculate breathing rate
        br = self.calculate_breathing_rate(signal_array)
        self.results['breathing_rate'] = int(br) if br > 0 else 0
        
        # Calculate HRV
        hrv = self.calculate_hrv(signal_array)
        self.results['hrv'] = int(hrv)
        
        # Calculate stress index
        stress = self.calculate_stress_index(hr, hrv, br)
        self.results['stress_index'] = round(stress, 2)
        
        # Calculate parasympathetic activity
        para = self.calculate_parasympathetic_activity(hrv, br)
        self.results['parasympathetic'] = int(para)
        
        # Estimate blood pressure
        sys_bp, dia_bp = self.estimate_blood_pressure(hr, hrv, stress)
        self.results['blood_pressure_sys'] = sys_bp
        self.results['blood_pressure_dia'] = dia_bp
        
       
        
        # Calculate wellness score
        wellness = self.calculate_wellness_score()
        self.results['wellness_score'] = int(wellness)
        
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
        
        # Other metrics
        self.result_labels['hrv'].config(text=f"{self.results['hrv']} ms")
        self.result_labels['stress_index'].config(text=f"{self.results['stress_index']}")
        self.result_labels['parasympathetic'].config(text=f"{self.results['parasympathetic']} %")
        self.result_labels['wellness_score'].config(text=f"{self.results['wellness_score']} /100")
    
    def start_monitoring(self):
        """Start health monitoring"""
        if not self.face_detected:
            messagebox.showwarning("Warning", "Please ensure your face is visible in the camera!")
            return
        
        self.monitoring_active = True
        self.ppg_signal.clear()
        self.timestamps.clear()
        self.hr_signal.clear()
        self.rr_signal.clear()
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
        
        self.progress_label.config(text="Starting monitoring...")
        messagebox.showinfo("Started", "Health monitoring started! Please remain still and look at the camera.")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        self.session_data['end_time'] = datetime.now().isoformat()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.NORMAL)
        
        self.progress['value'] = 0
        self.progress_label.config(text="Monitoring stopped. Data ready for export.")
        
        messagebox.showinfo("Stopped", "Monitoring stopped. You can now export your data.")
    
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
            'Duration (minutes)': [self.calculate_session_duration()],
            'Total Measurements': [len(self.session_data['measurements'])]
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
        
        # Write to CSV with multiple sheets simulation
        with open(filename, 'w', newline='') as f:
            f.write("=== HEALTH MONITORING SESSION SUMMARY ===\n")
            summary_df.to_csv(f, index=False)
            f.write("\n=== DETAILED MEASUREMENTS ===\n")
            measurements_df.to_csv(f, index=False)
            
            # Add raw PPG data if available
            if self.session_data['raw_ppg_data']:
                f.write("\n=== RAW PPG SIGNAL DATA ===\n")
                ppg_df = pd.DataFrame({
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
                'duration_minutes': self.calculate_session_duration(),
                'total_measurements': len(self.session_data['measurements'])
            },
            'final_results': self.results,
            'measurements_timeline': self.session_data['measurements'],
            'raw_ppg_data': {
                'timestamps': self.session_data['timestamps_data'][:len(self.session_data['raw_ppg_data'])],
                'values': self.session_data['raw_ppg_data']
            } if self.session_data['raw_ppg_data'] else None
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_to_excel(self, filename):
        """Export data to Excel format with multiple sheets"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': ['Session Start', 'Session End', 'Duration (minutes)', 'Total Measurements'],
                    'Value': [
                        self.session_data['start_time'],
                        self.session_data['end_time'],
                        self.calculate_session_duration(),
                        len(self.session_data['measurements'])
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
                
                # Raw PPG data
                if self.session_data['raw_ppg_data']:
                    ppg_df = pd.DataFrame({
                        'timestamp': self.session_data['timestamps_data'][:len(self.session_data['raw_ppg_data'])],
                        'ppg_value': self.session_data['raw_ppg_data']
                    })
                    ppg_df.to_excel(writer, sheet_name='Raw PPG Data', index=False)
                    
        except ImportError:
            # Fallback to CSV if openpyxl is not available
            messagebox.showwarning("Warning", "Excel export requires openpyxl. Saving as CSV instead.")
            self.export_to_csv(filename.replace('.xlsx', '.csv'))
    
    def calculate_session_duration(self):
        """Calculate session duration in minutes"""
        if not self.session_data['start_time'] or not self.session_data['end_time']:
            return 0
        
        try:
            start = datetime.fromisoformat(self.session_data['start_time'])
            end = datetime.fromisoformat(self.session_data['end_time'])
            duration = (end - start).total_seconds() / 60
            return round(duration, 2)
        except:
            return 0
    
    def generate_health_report(self):
        """Generate a comprehensive health report"""
        if not self.session_data['measurements']:
            return "No data available for report generation."
        
        report = f"""
HEALTH MONITORING REPORT
========================

Session Information:
- Start Time: {self.session_data['start_time']}
- End Time: {self.session_data['end_time']}
- Duration: {self.calculate_session_duration()} minutes
- Total Measurements: {len(self.session_data['measurements'])}

Final Measurements:
- Heart Rate: {self.results['heart_rate']} bpm
- Breathing Rate: {self.results['breathing_rate']} rpm
- Blood Pressure: {self.results['blood_pressure_sys']}/{self.results['blood_pressure_dia']} mmHg
- HRV: {self.results['hrv']} ms
- Stress Index: {self.results['stress_index']}
- Parasympathetic Activity: {self.results['parasympathetic']}%
- Wellness Score: {self.results['wellness_score']}/100

Health Interpretation:
{self.get_health_interpretation()}

Recommendations:
{self.get_recommendations()}
        """
        return report.strip()
    
    def get_health_interpretation(self):
        """Get health interpretation based on measurements"""
        interpretations = []
        
        # Heart Rate
        hr = self.results['heart_rate']
        if hr < 60:
            interpretations.append("• Heart Rate: Below normal (Bradycardia)")
        elif hr > 100:
            interpretations.append("• Heart Rate: Above normal (Tachycardia)")
        else:
            interpretations.append("• Heart Rate: Normal range")
        
        # Blood Pressure
        sys_bp = self.results['blood_pressure_sys']
        if sys_bp < 120:
            interpretations.append("• Blood Pressure: Normal")
        elif sys_bp < 140:
            interpretations.append("• Blood Pressure: Elevated")
        else:
            interpretations.append("• Blood Pressure: High")
        
        # Stress Level
        stress = self.results['stress_index']
        if stress < 0.3:
            interpretations.append("• Stress Level: Low")
        elif stress < 0.7:
            interpretations.append("• Stress Level: Moderate")
        else:
            interpretations.append("• Stress Level: High")
        
        # Wellness Score
        wellness = self.results['wellness_score']
        if wellness > 80:
            interpretations.append("• Overall Wellness: Excellent")
        elif wellness > 60:
            interpretations.append("• Overall Wellness: Good")
        elif wellness > 40:
            interpretations.append("• Overall Wellness: Fair")
        else:
            interpretations.append("• Overall Wellness: Poor")
        
        return '\n'.join(interpretations)
    
    def get_recommendations(self):
        """Get health recommendations based on measurements"""
        recommendations = []
        
        # Based on heart rate
        hr = self.results['heart_rate']
        if hr > 100:
            recommendations.append("• Consider relaxation techniques or consult a healthcare provider")
        elif hr < 60 and self.results['stress_index'] > 0.5:
            recommendations.append("• Monitor heart rate and consider medical consultation")
        
        # Based on stress index
        stress = self.results['stress_index']
        if stress > 0.7:
            recommendations.append("• Practice stress management techniques (meditation, deep breathing)")
            recommendations.append("• Ensure adequate sleep and regular exercise")
        
        # Based on breathing rate
        br = self.results['breathing_rate']
        if br > 20:
            recommendations.append("• Practice slow, deep breathing exercises")
        
        # Based on wellness score
        wellness = self.results['wellness_score']
        if wellness < 60:
            recommendations.append("• Consider lifestyle improvements (diet, exercise, sleep)")
            recommendations.append("• Regular health checkups recommended")
        
        # General recommendations
        recommendations.append("• Stay hydrated and maintain regular exercise")
        recommendations.append("• Maintain consistent sleep schedule")
        recommendations.append("• Consider regular health monitoring")
        
        return '\n'.join(recommendations) if recommendations else "• Continue maintaining healthy lifestyle habits"
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function"""
    try:
        app = FaceVitalMonitor()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()