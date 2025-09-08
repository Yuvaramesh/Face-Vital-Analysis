import cv2
import numpy as np
import mediapipe as mp
import time
from scipy import signal
from scipy.fft import fft
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import math

class FaceVitalMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Vital - Health Monitor (MediaPipe)")
        self.root.geometry("1200x800")
        
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
        self.face_detected = False
        self.monitoring_active = False
        
        # Results
        self.results = {
            'heart_rate': 0,
            'breathing_rate': 0,
            'blood_pressure_sys': 0,
            'blood_pressure_dia': 0,
            'bmi': 0,
            'hrv': 0,
            'stress_index': 0,
            'parasympathetic': 0,
            'wellness_score': 0
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
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
        
        # Progress bar
        self.progress = ttk.Progressbar(video_frame, length=300, mode='determinate')
        self.progress.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Health Metrics", padding="10")
        results_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create result labels
        self.result_labels = {}
        metrics = [
            ("Heart Rate", "heart_rate", "bpm"),
            ("Breathing Rate", "breathing_rate", "rpm"),
            ("Blood Pressure", "blood_pressure", "mmHg"),
            ("BMI", "bmi", ""),
            ("HRV", "hrv", "ms"),
            ("Stress Index", "stress_index", ""),
            ("Parasympathetic", "parasympathetic", "%"),
            ("Wellness Score", "wellness_score", "/100")
        ]
        
        for i, (name, key, unit) in enumerate(metrics):
            ttk.Label(results_frame, text=f"{name}:", font=('Arial', 10, 'bold')).grid(
                row=i, column=0, sticky='w', pady=5)
            
            if key == "blood_pressure":
                self.result_labels[key] = ttk.Label(results_frame, text="--/-- mmHg", 
                                                   font=('Arial', 12))
            else:
                self.result_labels[key] = ttk.Label(results_frame, text=f"-- {unit}", 
                                                   font=('Arial', 12))
            self.result_labels[key].grid(row=i, column=1, sticky='w', padx=10, pady=5)
        
        # Instructions
        instructions = ttk.Label(results_frame, 
                               text="Instructions:\n\n1. Sit comfortably\n2. Look directly at camera\n3. Keep face well-lit\n4. Minimize movement\n5. Wait for 30 seconds", 
                               font=('Arial', 9), justify=tk.LEFT)
        instructions.grid(row=len(metrics), column=0, columnspan=2, pady=20, sticky='w')
        
        # Start video feed
        self.update_video()
    
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
    
    def estimate_bmi(self, landmarks):
        """Estimate BMI using facial width-to-height ratio (very approximate)"""
        try:
            # Get face dimensions using MediaPipe landmarks
            # Face width (left to right)
            left_face = landmarks[234]  # Left face boundary
            right_face = landmarks[454]  # Right face boundary
            face_width = abs(right_face.x - left_face.x)
            
            # Face height (top to bottom)
            top_face = landmarks[10]   # Top of head
            bottom_face = landmarks[152]  # Chin
            face_height = abs(top_face.y - bottom_face.y)
            
            # Empirical correlation (this is highly approximate)
            face_ratio = face_width / face_height if face_height > 0 else 1.5
            
            # Very rough estimation - would need proper calibration
            estimated_bmi = 18 + (face_ratio - 0.8) * 20
            return max(15, min(40, estimated_bmi))
            
        except Exception as e:
            print(f"Error estimating BMI: {e}")
            return 22  # Average BMI
    
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
                
                # Update progress
                progress = min(100, (len(self.ppg_signal) / 900) * 100)
                self.progress['value'] = progress
                
                # Calculate metrics if we have enough data
                if len(self.ppg_signal) >= 300:  # 10 seconds minimum
                    self.calculate_all_metrics(face_landmarks.landmark)
        else:
            self.face_detected = False
            
        # Convert frame for display
        frame_pil = Image.fromarray(frame)
        frame_pil = frame_pil.resize((400, 300))
        frame_tk = ImageTk.PhotoImage(frame_pil)
        
        self.video_label.configure(image=frame_tk)
        self.video_label.image = frame_tk
        
        # Add status text
        status = "Face Detected" if self.face_detected else "No Face Detected"
        if self.monitoring_active:
            status += f" - Recording ({len(self.ppg_signal)}/900 frames)"
        
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
        
        # Estimate BMI
        bmi = self.estimate_bmi(landmarks)
        self.results['bmi'] = round(bmi, 1)
        
        # Calculate wellness score
        wellness = self.calculate_wellness_score()
        self.results['wellness_score'] = int(wellness)
        
        # Update UI
        self.update_results_display()
    
    def update_results_display(self):
        """Update the results display"""
        self.result_labels['heart_rate'].config(text=f"{self.results['heart_rate']} bpm")
        self.result_labels['breathing_rate'].config(text=f"{self.results['breathing_rate']} rpm")
        self.result_labels['blood_pressure'].config(
            text=f"{self.results['blood_pressure_sys']}/{self.results['blood_pressure_dia']} mmHg")
        self.result_labels['bmi'].config(text=f"{self.results['bmi']}")
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
        self.progress['value'] = 0
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        messagebox.showinfo("Started", "Health monitoring started! Please remain still and look at the camera.")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
    
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