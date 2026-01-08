import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageFilter
import cv2
import numpy as np
import json
import os
import glob
import re
import pytesseract
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import datetime
import sys

# --- CONFIGURATION ---
GAME_W = 1920
GAME_H = 1080
INPUT_FOLDER = "captured_data"
COORDS_FILE = "city_coordinates.json" 
OUTPUT_FILE = os.path.join(INPUT_FOLDER, "combined_dataset.json")
MODEL_FILE = "active_learning_model.keras"

ROI_SIZE = 70
IMG_SHAPE = (ROI_SIZE, ROI_SIZE, 3)
EPOCHS_INITIAL = 10
EPOCHS_UPDATE = 5

TOP_CUBES_Y = 12
BOT_CUBES_Y = 33

BAR_REGIONS = [
    ("player_cards", "Player Cards", (484, 20, 51, 38), "0123456789"),
    ("infection_cards", "Infection Cards", (743, 20, 41, 38), "0123456789"),
    ("infection_rate", "Infection Rate", (907, TOP_CUBES_Y, 29, BOT_CUBES_Y - TOP_CUBES_Y), "0123456789"),
    ("outbreaks", "Outbreaks", (1118, TOP_CUBES_Y, 26, BOT_CUBES_Y - TOP_CUBES_Y), "0123456789"),
    ("yellow_supply", "Yellow Supply", (1275, TOP_CUBES_Y, 32, BOT_CUBES_Y - TOP_CUBES_Y), "0123456789"),
    ("black_supply", "Black Supply", (1350, TOP_CUBES_Y, 28, BOT_CUBES_Y - TOP_CUBES_Y), "0123456789"),
    ("red_supply", "Red Supply", (1421, TOP_CUBES_Y, 32, BOT_CUBES_Y - TOP_CUBES_Y), "0123456789"),
    ("blue_supply", "Blue Supply", (1491, TOP_CUBES_Y, 31, BOT_CUBES_Y - TOP_CUBES_Y), "0123456789"),
]

ACTIONS_REGION = {
    "key": "actions_taken",
    "label": "Actions Taken",
    "region": (675, 980, 730 - 675, 1050 - 980),
    "whitelist": "01234/"
}

COLORS_HSV = {
    "yellow": ((20, 150, 150), (40, 255, 255)),
    "red":    ((0, 150, 150), (10, 255, 255)),
    "blue":   ((100, 150, 150), (130, 255, 255)),
    "gray":   ((0, 0, 0), (180, 255, 60)) 
}

# --- LOGGING HELPER ---
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

# --- KERAS CALLBACK FOR CLEAN LOGS ---
class VerboseLogger(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0)
        mae = logs.get('mae', 0)
        log(f"Epoch {epoch+1}: loss={loss:.5f}, mae={mae:.5f}")

# --- HELPERS ---

def add_border(pil_img, border_size=10, color=0):
    w, h = pil_img.size
    new_img = Image.new("L", (w + 2*border_size, h + 2*border_size), color)
    new_img.paste(pil_img, (border_size, border_size))
    return new_img

def preprocess_image_ocr(img):
    UPSCALE = 3
    THRESHOLD = 160
    img = img.convert("L")
    w, h = img.size
    img = img.resize((max(1, w * UPSCALE), max(1, h * UPSCALE)), resample=Image.NEAREST)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda p: 255 if p > THRESHOLD else 0)
    return img

def run_smart_ocr(pil_crop, whitelist="0123456789"):
    crop_rgb = pil_crop.convert("RGB")
    cw, ch = crop_rgb.size
    pixels = crop_rgb.load()
    
    for px in range(cw):
        for py in range(ch):
            r, g, b = pixels[px, py]
            is_white = (r > 200 and g > 200 and b > 200)
            is_gray = (pixels[px, py] == (162, 163, 163))
            is_red = (r > 130 and g < 100 and b < 100)
            
            if is_white or is_gray or is_red:
                pixels[px, py] = (255, 255, 255)
            else:
                pixels[px, py] = (0, 0, 0)
    
    proc_img = preprocess_image_ocr(crop_rgb)
    
    variants = {}
    try:
        variants["orig"] = proc_img
        variants["stretched"] = proc_img.resize((int(proc_img.width * 1.5), proc_img.height), resample=Image.NEAREST)
        variants["inverted"] = ImageOps.invert(proc_img)
    except:
        variants = {"orig": proc_img}

    base_cfg = f"-c tessedit_char_whitelist={whitelist}"
    configs = [
        f"{base_cfg} --psm 13",
        f"{base_cfg} --psm 7",
        f"{base_cfg} --psm 6",
        f"{base_cfg} --psm 10",
    ]

    for name, vimg in variants.items():
        pad_col = 255 if "inverted" in name else 0
        vimg_padded = add_border(vimg, 10, pad_col)
        for cfg in configs:
            try:
                raw = pytesseract.image_to_string(vimg_padded, config=cfg).strip()
                clean = raw.replace(" ", "")
                if clean: return clean
            except: pass
    return ""

def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4)) 
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

class CombinedLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Pandemic AI - Active Learning Labeler")
        self.root.geometry("1600x900")
        
        self.groups = {} 
        self.sorted_timestamps = []
        self.current_idx = 0
        self.current_timestamp = None
        self.current_file_list = []
        
        self.city_coords = {}
        self.city_data = {} 
        self.bar_data_vars = {} 
        self.supply_labels = {} 
        
        self.tool_color = "yellow"
        self.tool_value = 1
        self.display_colors = {"yellow": "#ffff00", "red": "#ff4444", "blue": "#4444ff", "gray": "#aaaaaa"}

        self.model = None
        self.is_model_trained = False

        if os.path.exists(COORDS_FILE):
            with open(COORDS_FILE, 'r') as f:
                self.city_coords = json.load(f)
        
        self.scan_directory()
        self.filter_existing_data()
        
        if os.path.exists(MODEL_FILE):
            log(f"Loading existing AI model from {MODEL_FILE}...")
            try:
                self.model = tf.keras.models.load_model(MODEL_FILE)
                self.is_model_trained = True
                log("Model loaded successfully.")
            except Exception as e:
                log(f"Error loading model: {e}. Will create new one.")
                
        if not self.is_model_trained and os.path.exists(OUTPUT_FILE):
            log("No saved model found, but dataset exists. Training initial model...")
            self.train_ai_model(initial=True)
        elif not self.is_model_trained:
            log("No model and no dataset. Starting in Heuristic mode.")

        if not self.sorted_timestamps:
            messagebox.showinfo("Done", "All groups processed!")
            root.destroy()
            return

        self.setup_ui()
        self.load_group(0)

    def train_ai_model(self, initial=False):
        if not os.path.exists(OUTPUT_FILE): return
        with open(OUTPUT_FILE, 'r') as f: dataset = json.load(f)
        if len(dataset) < 5:
            log("Not enough data to train yet.")
            return

        X, y = [], []
        log(f"Preparing training data from {len(dataset)} records...")
        
        data_slice = dataset if initial else dataset[-500:] 
        
        for entry in data_slice:
            if not os.path.exists(entry['image']): continue
            full_img = cv2.imread(entry['image'])
            if full_img is None: continue
            city_labels = entry['city_infections']
            
            for city_name, coords in self.city_coords.items():
                if city_name not in city_labels: continue
                cx, cy = int(coords[0]), int(coords[1])
                half = ROI_SIZE // 2
                y1, y2 = max(0, cy - half), min(GAME_H, cy + half)
                x1, x2 = max(0, cx - half), min(GAME_W, cx + half)
                crop = full_img[y1:y2, x1:x2]
                if crop.shape[0] != ROI_SIZE or crop.shape[1] != ROI_SIZE:
                    crop = cv2.resize(crop, (ROI_SIZE, ROI_SIZE))
                crop = crop.astype('float32') / 255.0
                c = city_labels[city_name]
                X.append(crop)
                y.append([c['yellow'], c['red'], c['blue'], c['gray']])
        
        if not X: return
        if self.model is None: self.model = create_cnn_model()
        
        ep = EPOCHS_INITIAL if initial else EPOCHS_UPDATE
        log(f"Training model on {len(X)} samples for {ep} epochs...")
        
        # Use custom verbose logger to avoid chaos
        self.model.fit(np.array(X), np.array(y), epochs=ep, batch_size=32, verbose=0, callbacks=[VerboseLogger()])
        self.model.save(MODEL_FILE)
        self.is_model_trained = True
        log("Training complete.")

    def run_ai_prediction(self):
        log("Running AI Prediction...")
        full_img_bgr = cv2.cvtColor(np.array(self.pil_img), cv2.COLOR_RGB2BGR)
        rois, names = [], []
        
        for name, coords in self.city_coords.items():
            cx, cy = int(coords[0]), int(coords[1])
            half = ROI_SIZE // 2
            y1, y2 = max(0, cy - half), min(GAME_H, cy + half)
            x1, x2 = max(0, cx - half), min(GAME_W, cx + half)
            crop = full_img_bgr[y1:y2, x1:x2]
            if crop.shape[0] != ROI_SIZE or crop.shape[1] != ROI_SIZE:
                crop = cv2.resize(crop, (ROI_SIZE, ROI_SIZE))
            rois.append(crop.astype('float32') / 255.0)
            names.append(name)
            
        preds = self.model.predict(np.array(rois), verbose=0)
        for i, name in enumerate(names):
            cnt = np.maximum(np.round(preds[i]).astype(int), 0)
            self.city_data[name] = {"yellow": int(cnt[0]), "red": int(cnt[1]), "blue": int(cnt[2]), "gray": int(cnt[3])}

    def scan_directory(self):
        log("Scanning input folder...")
        if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)
        files = glob.glob(os.path.join(INPUT_FOLDER, "*.png"))
        self.groups = {}
        for fpath in files:
            fname = os.path.basename(fpath)
            match = re.search(r"screen_(\d{8}_\d{6})_", fname)
            if match:
                ts = match.group(1)
                if ts not in self.groups: self.groups[ts] = []
                self.groups[ts].append(fpath)
        self.sorted_timestamps = sorted(self.groups.keys())

    def filter_existing_data(self):
        if not os.path.exists(OUTPUT_FILE): return
        try:
            with open(OUTPUT_FILE, 'r') as f: existing = json.load(f)
            processed_ts = set(item.get('timestamp') for item in existing)
            to_remove = [ts for ts in self.groups if ts in processed_ts]
            for ts in to_remove: del self.groups[ts]
            self.sorted_timestamps = sorted(self.groups.keys())
        except: pass

    def setup_ui(self):
        sidebar_outer = tk.Frame(self.root, width=420, bg="#ddd")
        sidebar_outer.pack(side=tk.RIGHT, fill=tk.Y)
        sb_canvas = tk.Canvas(sidebar_outer, bg="#ddd")
        scrollbar = tk.Scrollbar(sidebar_outer, orient="vertical", command=sb_canvas.yview)
        self.sidebar = tk.Frame(sb_canvas, bg="#ddd")
        self.sidebar.bind("<Configure>", lambda e: sb_canvas.configure(scrollregion=sb_canvas.bbox("all")))
        sb_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        sb_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        map_frame = tk.Frame(self.root, bg="#222")
        map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.map_canvas = tk.Canvas(map_frame, bg="#222")
        self.map_canvas.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.sidebar, text="--- BATCH CONTROL ---", font=("Arial", 11, "bold"), bg="#ddd").pack(pady=5)
        self.lbl_info = tk.Label(self.sidebar, text="...", bg="#ddd")
        self.lbl_info.pack()
        
        nav_frame = tk.Frame(self.sidebar, bg="#ddd")
        nav_frame.pack(pady=5)
        tk.Button(nav_frame, text="Skip", command=self.next_group).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="SAVE & RETRAIN", command=self.save_batch, bg="#00cc00", fg="white", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=5)

        tk.Label(self.sidebar, text="--- MAP PAINT TOOL ---", font=("Arial", 11, "bold"), bg="#ddd").pack(pady=5)
        self.lbl_tool = tk.Label(self.sidebar, text="Tool: YELLOW [1]", font=("Arial", 12, "bold"), fg="#aaaa00", bg="#ddd")
        self.lbl_tool.pack(pady=5)

        val_frame = tk.Frame(self.sidebar, bg="#ddd")
        val_frame.pack(pady=2)
        tk.Button(val_frame, text="1", width=3, command=lambda: self.set_val(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(val_frame, text="2", width=3, command=lambda: self.set_val(2)).pack(side=tk.LEFT, padx=2)
        tk.Button(val_frame, text="3", width=3, command=lambda: self.set_val(3)).pack(side=tk.LEFT, padx=2)
        tk.Button(val_frame, text="0", width=3, command=lambda: self.set_val(0)).pack(side=tk.LEFT, padx=2)

        clear_frame = tk.Frame(self.sidebar, bg="#ddd")
        clear_frame.pack(pady=5)
        tk.Button(clear_frame, text=" Y ", bg="#ffff00", fg="black", command=lambda: self.clear_color('yellow')).pack(side=tk.LEFT, padx=2)
        tk.Button(clear_frame, text=" R ", bg="#ff4444", fg="white", command=lambda: self.clear_color('red')).pack(side=tk.LEFT, padx=2)
        tk.Button(clear_frame, text=" B ", bg="#4444ff", fg="white", command=lambda: self.clear_color('blue')).pack(side=tk.LEFT, padx=2)
        tk.Button(clear_frame, text=" G ", bg="#666666", fg="white", command=lambda: self.clear_color('gray')).pack(side=tk.LEFT, padx=2)

        tk.Label(self.sidebar, text="--- SUPPLY MONITOR ---", font=("Arial", 11, "bold"), bg="#ddd").pack(pady=5)
        supply_frame = tk.Frame(self.sidebar, bg="#eee", bd=1, relief=tk.SOLID)
        supply_frame.pack(fill=tk.X, padx=5, pady=5)
        for i, color in enumerate(["yellow", "red", "blue", "gray"]):
            lbl = tk.Label(supply_frame, text=f"{color.title()}: 0 / 0", font=("Consolas", 11, "bold"), bg="#eee")
            lbl.grid(row=i//2, column=i%2, sticky="w", padx=10, pady=2)
            self.supply_labels[color] = lbl

        tk.Label(self.sidebar, text="--- SCREEN VALUES ---", font=("Arial", 11, "bold"), bg="#ddd").pack(pady=10)
        self.crop_labels = {} 
        self.create_entry_row(ACTIONS_REGION["key"], ACTIONS_REGION["label"])
        for key, label, _, _ in BAR_REGIONS:
            self.create_entry_row(key, label)

        self.map_canvas.bind("<Button-1>", self.on_map_click)
        self.map_canvas.bind("<Button-3>", self.on_map_right_click)
        self.map_canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.root.bind('<Key>', self.handle_keypress)

        self.cursor_id = self.map_canvas.create_text(0,0, text="1", fill="yellow", font=("Arial",20,"bold"), state='hidden')
        self.cursor_bg = self.map_canvas.create_rectangle(0,0,0,0, fill="black", state='hidden')
        self.map_canvas.tag_lower(self.cursor_bg, self.cursor_id)

    def create_entry_row(self, key, label_text):
        row = tk.Frame(self.sidebar, bg="#ddd", pady=2)
        row.pack(fill=tk.X, padx=5)
        tk.Label(row, text=label_text, width=15, anchor="w", bg="#ddd").pack(side=tk.LEFT)
        img_lbl = tk.Label(row, bg="#aaa", width=50, height=25)
        img_lbl.pack(side=tk.LEFT, padx=5)
        self.crop_labels[key] = img_lbl
        
        var = tk.StringVar()
        var.trace("w", lambda name, index, mode, v=var: self.update_supply_monitor()) 
        self.bar_data_vars[key] = var
        e = tk.Entry(row, textvariable=var, width=8, font=("Arial", 11))
        e.pack(side=tk.RIGHT)

    def handle_keypress(self, event):
        if isinstance(self.root.focus_get(), tk.Entry): return
        k = event.char.lower()
        if k == 'y': self.set_tool('yellow')
        elif k == 'r': self.set_tool('red')
        elif k == 'b': self.set_tool('blue')
        elif k == 'g': self.set_tool('gray')
        elif k in ['1', '2', '3', '0']: self.set_val(int(k))

    def _on_mousewheel(self, event):
        self.map_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_shift_mousewheel(self, event):
        self.map_canvas.xview_scroll(int(-1*(event.delta/120)), "units")

    def load_group(self, idx):
        if idx >= len(self.sorted_timestamps):
            messagebox.showinfo("Done", "All groups processed.")
            self.root.destroy()
            return

        self.current_idx = idx
        self.current_timestamp = self.sorted_timestamps[idx]
        self.current_file_list = self.groups[self.current_timestamp]
        self.lbl_info.config(text=f"Group {idx+1}/{len(self.sorted_timestamps)}\nTS: {self.current_timestamp}\nImages: {len(self.current_file_list)}")
        
        img_path = self.current_file_list[0]
        self.cv_img = cv2.imread(img_path)
        self.pil_img = Image.open(img_path)
        self.tk_map_img = ImageTk.PhotoImage(self.pil_img)
        self.map_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_map_img)
        self.map_canvas.config(scrollregion=(0,0, GAME_W, GAME_H))
        
        self.city_data = {}
        for city in self.city_coords:
            self.city_data[city] = {"yellow": 0, "red": 0, "blue": 0, "gray": 0}
            
        if self.is_model_trained:
            self.run_ai_prediction()
        else:
            self.run_heuristics()
            
        self.run_ocr_bars()
        self.redraw_map_overlays()
        self.update_supply_monitor()

    def run_heuristics(self):
        log("Running Heuristic Detection...")
        hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
        box_r = 35 
        for name, (cx, cy) in self.city_coords.items():
            cx, cy = int(cx), int(cy)
            roi = hsv[max(0, cy-box_r):min(GAME_H, cy+box_r), max(0, cx-box_r):min(GAME_W, cx+box_r)]
            if roi.size == 0: continue
            for color_name, (lower, upper) in COLORS_HSV.items():
                mask = cv2.inRange(roi, np.array(lower), np.array(upper))
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                count = sum(1 for c in cnts if cv2.contourArea(c) > 30)
                self.city_data[name][color_name] = min(count, 3)

    def run_ocr_bars(self):
        for key, _, rect, whitelist in BAR_REGIONS:
            x, y, w, h = rect
            crop = self.pil_img.crop((x, y, x+w, y+h))
            disp = crop.resize((w*2, h*2), Image.NEAREST)
            tk_disp = ImageTk.PhotoImage(disp)
            self.crop_labels[key].config(image=tk_disp, width=w*2, height=h*2)
            self.crop_labels[key].image = tk_disp
            res = run_smart_ocr(crop, whitelist)
            self.bar_data_vars[key].set(res)

        key = ACTIONS_REGION["key"]
        x, y, w, h = ACTIONS_REGION["region"]
        crop = self.pil_img.crop((x, y, x+w, y+h))
        disp = crop.resize((w, h), Image.NEAREST)
        tk_disp = ImageTk.PhotoImage(disp)
        self.crop_labels[key].config(image=tk_disp, width=w, height=h)
        self.crop_labels[key].image = tk_disp
        
        res = run_smart_ocr(crop, ACTIONS_REGION["whitelist"])
        match = re.search(r"(\d)", res)
        if match:
            self.bar_data_vars[key].set(match.group(1))
        else:
            self.bar_data_vars[key].set(res)

    def update_supply_monitor(self):
        map_totals = {"yellow": 0, "red": 0, "blue": 0, "gray": 0}
        for city_counts in self.city_data.values():
            for color in map_totals:
                map_totals[color] += city_counts[color]

        ocr_keys = {"yellow": "yellow_supply", "red": "red_supply", "blue": "blue_supply", "gray": "black_supply"}
        
        for color, total_on_map in map_totals.items():
            ocr_var = self.bar_data_vars[ocr_keys[color]]
            val_str = ocr_var.get().strip()
            target_str = "?"
            fg_color = "black"
            if val_str.isdigit():
                supply_left = int(val_str)
                expected_on_map = 24 - supply_left
                target_str = str(expected_on_map)
                fg_color = "#00aa00" if total_on_map == expected_on_map else "red"
            
            self.supply_labels[color].config(text=f"{color.title()}: {total_on_map} / {target_str}", fg=fg_color)

    def set_tool(self, color):
        self.tool_color = color
        self.update_tool_ui()
    
    def set_val(self, val):
        self.tool_value = val
        self.update_tool_ui()
        
    def update_tool_ui(self):
        hex_c = self.display_colors[self.tool_color]
        self.lbl_tool.config(text=f"Tool: {self.tool_color.upper()} [{self.tool_value}]", fg=hex_c)
        self.map_canvas.itemconfig(self.cursor_id, text=str(self.tool_value), fill=hex_c)

    def clear_color(self, color):
        for city in self.city_data:
            self.city_data[city][color] = 0
        self.redraw_map_overlays()
        self.update_supply_monitor()

    def on_mouse_move(self, event):
        cx = self.map_canvas.canvasx(event.x)
        cy = self.map_canvas.canvasy(event.y)
        self.map_canvas.coords(self.cursor_id, cx+15, cy-15)
        bbox = self.map_canvas.bbox(self.cursor_id)
        if bbox: self.map_canvas.coords(self.cursor_bg, bbox)
        self.map_canvas.itemconfig(self.cursor_id, state='normal')
        self.map_canvas.itemconfig(self.cursor_bg, state='normal')

    def on_map_click(self, event):
        self.map_canvas.focus_set()
        cx = self.map_canvas.canvasx(event.x)
        cy = self.map_canvas.canvasy(event.y)
        city = self.find_nearest_city(cx, cy)
        if city:
            self.city_data[city][self.tool_color] = self.tool_value
            self.redraw_map_overlays()
            self.update_supply_monitor()

    def on_map_right_click(self, event):
        self.map_canvas.focus_set()
        cx = self.map_canvas.canvasx(event.x)
        cy = self.map_canvas.canvasy(event.y)
        city = self.find_nearest_city(cx, cy)
        if city:
            for c in self.city_data[city]: self.city_data[city][c] = 0
            self.redraw_map_overlays()
            self.update_supply_monitor()

    def find_nearest_city(self, x, y):
        nearest = None; min_dist = 50
        for name, coords in self.city_coords.items():
            d = np.sqrt((coords[0]-x)**2 + (coords[1]-y)**2)
            if d < min_dist:
                min_dist = d; nearest = name
        return nearest

    def redraw_map_overlays(self):
        self.map_canvas.delete("overlay")
        for name, (cx, cy) in self.city_coords.items():
            self.map_canvas.create_oval(cx-5, cy-5, cx+5, cy+5, fill="white", tags="overlay")
            info = [f"{k[0].upper()}:{v}" for k,v in self.city_data[name].items() if v > 0]
            if info:
                txt = " ".join(info)
                self.map_canvas.create_rectangle(cx-25, cy-25, cx+25, cy-10, fill="black", stipple="gray50", tags="overlay")
                self.map_canvas.create_text(cx, cy-18, text=txt, fill="#00ff00", font=("Arial", 10, "bold"), tags="overlay")
        self.map_canvas.tag_raise(self.cursor_bg)
        self.map_canvas.tag_raise(self.cursor_id)

    def save_batch(self):
        bar_data = {}
        for key, var in self.bar_data_vars.items():
            val = var.get().strip()
            if key == "actions_taken":
                if val.isdigit(): bar_data[key] = int(val)
                else: bar_data[key] = None
            else:
                if val.isdigit(): bar_data[key] = int(val)
                else: bar_data[key] = None

        new_entries = []
        for fpath in self.current_file_list:
            entry = {
                "image": fpath,
                "timestamp": self.current_timestamp,
                "city_infections": self.city_data,
                "game_state": bar_data
            }
            new_entries.append(entry)
            
        full_data = []
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r') as f: full_data = json.load(f)
            except:
                pass
            
        full_data.extend(new_entries)
        with open(OUTPUT_FILE, 'w') as f: json.dump(full_data, f, indent=2)
        
        log(f"Saved {len(new_entries)} images. Retraining model...")
        self.root.update()
        self.train_ai_model(initial=False)
        self.next_group()

    def next_group(self):
        self.load_group(self.current_idx + 1)

if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        print("Captured data folder not found.")
    else:
        root = tk.Tk()
        app = CombinedLabeler(root)
        root.mainloop()
