import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import Toplevel, BooleanVar, Checkbutton
import cv2
import os
import shutil
import subprocess
import threading
import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def choose_input_file():
    path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi")])
    if path:
        input_var.set(path)

def run_pipeline():
    input_path = input_var.get()

    if not input_path:
        messagebox.showerror("Error", "Please select a video file first.")
        return

    log_area.delete(1.0, tk.END)
    log_area.insert(tk.END, "Starting processing pipeline...\n")

    def process():
        try:
            output_dir = "output"
            ilastik_path = ""  # Update if needed

            # Step 1: Clear/Create output folder
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            log_area.insert(tk.END, "🔹 Cleared and created output folder.\n")

            # Step 2: Extract frames
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = int(total_frames / fps)

            for sec in range(duration):
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                success, frame = cap.read()
                if success:
                    cv2.imwrite(f"{output_dir}/frame_{sec}.jpg", frame)

            cap.release()
            log_area.insert(tk.END, f"🔹 Extracted {duration} frames to output folder.\n")

            # Step 3: Launch ilastik GUI (user performs prediction manually)
            log_area.insert(tk.END, "🔹 Launching ilastik... (Close ilastik to continue)\n")
            subprocess.run([ilastik_path])
            log_area.insert(tk.END, "🔹 ilastik closed. Proceeding with watershed segmentation...\n")

            # Step 4: Apply Watershed on all images in output folder
            for filename in os.listdir(output_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                    path = os.path.join(output_dir, filename)
                    image = cv2.imread(path)
                    if image is None:
                        continue

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                    kernel = np.ones((4, 4), np.uint8)
                    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

                    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
                    sure_fg = np.uint8(sure_fg)
                    sure_bg = cv2.dilate(opening, kernel, iterations=3)
                    unknown = cv2.subtract(sure_bg, sure_fg)

                    _, markers = cv2.connectedComponents(sure_fg)
                    markers = markers + 1
                    markers[unknown == 255] = 0

                    markers = watershed(-dist_transform, markers, mask=opening)

                    segmented = np.zeros_like(gray)
                    for label in np.unique(markers):
                        if label <= 1:
                            continue
                        segmented[markers == label] = 255

                    cv2.imwrite(path, segmented)

            log_area.insert(tk.END, f"✅ Segmentation completed and saved in 'output/' folder.\n")
            messagebox.showinfo("Done", "✅ Processing pipeline completed!")

        except Exception as e:
            log_area.insert(tk.END, f"❌ Error: {e}\n")
            messagebox.showerror("Error", str(e))

    threading.Thread(target=process).start()

# GUI Setup
root = tk.Tk()
root.title("Video to Cell Segmentation Pipeline")
root.geometry("600x400")
root.configure(bg="#f0f0f0")

tk.Label(root, text="🧬 Cell Segmentation Pipeline", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=15)

input_var = tk.StringVar()
frame1 = tk.Frame(root, bg="#f0f0f0")
frame1.pack(pady=5)
tk.Label(frame1, text="Select Input Video: ", bg="#f0f0f0").pack(side=tk.LEFT)
tk.Entry(frame1, textvariable=input_var, width=40).pack(side=tk.LEFT)
tk.Button(frame1, text="📂 Browse", command=choose_input_file).pack(side=tk.LEFT, padx=5)

tk.Button(root, text="▶ Run Pipeline", command=run_pipeline, bg="#4285f4", fg="white",
          font=("Helvetica", 12, "bold"), padx=10, pady=5).pack(pady=15)

log_area = scrolledtext.ScrolledText(root, height=10, width=70)
log_area.pack(pady=10)
log_area.insert(tk.END, "Log Output will appear here...\n")

root.mainloop()


def choose_ilastik_features():
    feature_window = Toplevel(root)
    feature_window.title("Select ilastik Features")

    features = {
        "Gaussian Smoothing": BooleanVar(),
        "Laplacian of Gaussian": BooleanVar(),
        "Gaussian Gradient Magnitude": BooleanVar(),
        "Difference of Gaussians": BooleanVar(),
        "Structure Tensor Eigenvalues": BooleanVar(),
        "Hessian of Gaussian Eigenvalues": BooleanVar()
    }

    sigmas = {
        "0.30": BooleanVar(),
        "0.70": BooleanVar(),
        "1.00": BooleanVar(),
        "1.60": BooleanVar(),
        "3.50": BooleanVar(),
        "5.00": BooleanVar(),
        "10.00": BooleanVar()
    }

    tk.Label(feature_window, text="Features:").pack(anchor='w', padx=10)
    for f, var in features.items():
        Checkbutton(feature_window, text=f, variable=var).pack(anchor='w', padx=20)

    tk.Label(feature_window, text="Sigma Values:").pack(anchor='w', padx=10, pady=(10, 0))
    for s, var in sigmas.items():
        Checkbutton(feature_window, text=s, variable=var).pack(anchor='w', padx=20)

    def apply_selection():
        selected_features = [f for f, v in features.items() if v.get()]
        selected_sigmas = [float(s) for s, v in sigmas.items() if v.get()]
        print("Selected Features:", selected_features)
        print("Selected Sigmas:", selected_sigmas)

        # You can store these in a global or class variable to use in ilastik headless call
        feature_window.destroy()

    tk.Button(feature_window, text="Apply", command=apply_selection).pack(pady=10)
