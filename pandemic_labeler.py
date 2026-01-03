import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageFilter
import cv2
import numpy as np
import json
import os
import pyautogui
import datetime
import pytesseract
import time

# --- CONFIGURATION ---
GAME_W = 1920
GAME_H = 1080
OUTPUT_FOLDER = "captured_data"
COORDS_FILE = "city_coordinates.json" 
DATASET_FILE = os.path.join(OUTPUT_FOLDER, "pandemic_dataset.json")

# Debug settings for OCR
SAVE_DEBUG_IMAGES = True
DEBUG_FOLDER = "ocr_debug_steps"

if SAVE_DEBUG_IMAGES and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

# Burst Capture Settings
BURST_COUNT = 20
BURST_INTERVAL = 0.1 

# Point to tesseract exe if needed (uncomment if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# OCR Regions
TOP_INFECTION_CUBES = 12
BOTTOM_INFECTION_CUBES = 33
OCR_REGIONS = [
    (484, 20, 535 - 484, 58 - 20),
    (743, 20, 784 - 743, 58 - 20),
    (907, TOP_INFECTION_CUBES, 936 - 907, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),
    (1118, TOP_INFECTION_CUBES, 1144 - 1118, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES),
    (1275, TOP_INFECTION_CUBES, 1307 - 1275, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES), # 4: Yellow
    (1350, TOP_INFECTION_CUBES, 1378 - 1350, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES), # 5: Gray
    (1421, TOP_INFECTION_CUBES, 1453 - 1421, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES), # 6: Red
    (1491, TOP_INFECTION_CUBES, 1522 - 1491, BOTTOM_INFECTION_CUBES - TOP_INFECTION_CUBES), # 7: Blue
]

# HSV Color Ranges (for Heuristic)
COLORS_HSV = {
    "yellow": ((20, 150, 150), (40, 255, 255)),
    "red":    ((0, 150, 150), (10, 255, 255)),
    "blue":   ((100, 150, 150), (130, 255, 255)),
    "gray":   ((0, 0, 0), (180, 255, 60)) 
}

# --- OCR HELPERS ---
def preprocess_image_ocr(img):
    """
    Standard preprocessing: Grayscale -> Upscale -> Sharpen -> Binary Threshold.
    """
    UPSCALE = 3
    THRESHOLD = 160
    img = img.convert("L")
    w, h = img.size
    img = img.resize((max(1, w * UPSCALE), max(1, h * UPSCALE)), resample=Image.NEAREST)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda p: 255 if p > THRESHOLD else 0)
    return img

def try_ocr_best(pil_img, region_idx):
    """
    Try several preprocessing variants and tesseract configs.
    Includes horizontal stretching and PSM 13 to separate close digits.
    Returns integer if found, else None.
    """
    base = pil_img.convert("L")
    variants = {}
    
    try:
        variants["orig"] = base
        
        # Add slight horizontal stretch to separate digits if they are too close (Fix for 22 -> 2)
        variants["stretched"] = base.resize((int(base.width * 1.5), base.height), resample=Image.NEAREST)
        
        variants["inverted"] = ImageOps.invert(base)
        
        res_x4 = base.resize((base.width*4, base.height*4), resample=Image.NEAREST)
        variants["res_x4"] = res_x4
        
        # Stretch the upscaled version as well
        variants["res_x4_stretched"] = res_x4.resize((int(res_x4.width * 1.2), res_x4.height), resample=Image.NEAREST)
        
        variants["res_x4_inverted"] = ImageOps.invert(res_x4)
        variants["res_x4_dilated"] = res_x4.filter(ImageFilter.MaxFilter(3))
        
        variants["autocontrast"] = ImageOps.autocontrast(base)
        variants["autocontrast_res_x4"] = ImageOps.autocontrast(res_x4)
    except Exception:
        variants = {"orig": base}

    # Tesseract configurations
    configs = [
        r'-c tessedit_char_whitelist=0123456789 --psm 13', # Raw line (Best for "22")
        r'-c tessedit_char_whitelist=0123456789 --psm 7',  # Single line
        r'-c tessedit_char_whitelist=0123456789 --psm 8',  # Single word
        r'-c tessedit_char_whitelist=0123456789 --psm 6',  # Block of text
        r'-c tessedit_char_whitelist=0123456789 --psm 10', # Single char (fallback)
    ]

    for name, vimg in variants.items():
        if SAVE_DEBUG_IMAGES:
            filename = f"labeler_region_{region_idx}_variant_{name}.png"
            vimg.save(os.path.join(DEBUG_FOLDER, filename))

        for cfg in configs:
            try:
                raw = pytesseract.image_to_string(vimg, config=cfg)
                digits = ''.join(ch for ch in raw if ch.isdigit())
                if digits: 
                    return int(digits)
            except: 
                pass
    return None

class PandemicLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Pandemic AI Trainer - PAINT MODE")
        self.root.geometry("1200x900")

        # Capture Burst
        self.burst_paths = self.take_burst_screenshots()
        self.image_path = self.burst_paths[0]
        
        self.cv_image = cv2.imread(self.image_path)
        self.pil_image = Image.open(self.image_path).resize((GAME_W, GAME_H))
        self.tk_image = ImageTk.PhotoImage(self.pil_image)

        # State
        self.cities = {} 
        self.data = {}    
        self.setup_mode = False
        self.ocr_targets = {"yellow": 24, "red": 24, "blue": 24, "gray": 24}
        
        # PAINT TOOL STATE
        self.tool_color = "yellow"
        self.tool_value = 1
        self.display_colors = {"yellow": "#ffff00", "red": "#ff4444", "blue": "#4444ff", "gray": "#aaaaaa"}
        
        # Load coords
        if os.path.exists(COORDS_FILE):
            with open(COORDS_FILE, 'r') as f:
                self.cities = json.load(f)
        
        self.reset_data()
        
        # Run OCR with the new manual thresholding logic
        self.run_ocr_supply()

        # --- GUI LAYOUT ---
        top_container = tk.Frame(root, bg="#333", bd=2, relief=tk.RAISED)
        top_container.pack(side=tk.TOP, fill=tk.X)

        # Instructions / Tool Status
        self.info_label = tk.Label(
            top_container, 
            text="KEYS: [Y]ellow [R]ed [B]lue [G]ray | [1] [2] [3] [0]", 
            fg="white", bg="#333", font=("Arial", 11)
        )
        self.info_label.pack(side=tk.TOP, pady=5)

        # Controls
        control_frame = tk.Frame(top_container, bg="#444", pady=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_setup = tk.Button(control_frame, text="1. Setup Mode (OFF)", bg="lightgray", command=self.toggle_setup)
        self.btn_setup.pack(side=tk.LEFT, padx=10)
        tk.Button(control_frame, text="2. Auto-Detect", command=self.run_heuristic).pack(side=tk.LEFT, padx=10)
        tk.Button(control_frame, text="3. SAVE BATCH (20x)", command=self.save_dataset, bg="#00cc00", fg="white", font=("Arial",10,"bold")).pack(side=tk.RIGHT, padx=10)
        
        # Stats
        self.stats_frame = tk.Frame(top_container, bg="#222", pady=5)
        self.stats_frame.pack(side=tk.TOP, fill=tk.X)
        self.stats_labels = {}
        for color in ["yellow", "red", "blue", "gray"]:
            lbl = tk.Label(self.stats_frame, text="...", fg=self.display_colors[color], bg="#222", font=("Consolas", 12, "bold"))
            lbl.pack(side=tk.LEFT, expand=True)
            self.stats_labels[color] = lbl

        # Canvas
        canvas_container = tk.Frame(root)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        v_scroll = tk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        h_scroll = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        
        self.canvas = tk.Canvas(canvas_container, width=1000, height=700, bg="#222",
                                xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        v_scroll.config(command=self.canvas.yview)
        h_scroll.config(command=self.canvas.xview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, GAME_W, GAME_H))

        # --- EVENTS & BINDINGS ---
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        
        # Mouse movement for custom cursor
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)
        self.canvas.bind("<Enter>", self.on_mouse_enter)

        # Keyboard shortcuts
        self.root.bind('y', lambda e: self.set_tool_color('yellow'))
        self.root.bind('r', lambda e: self.set_tool_color('red'))
        self.root.bind('b', lambda e: self.set_tool_color('blue'))
        self.root.bind('g', lambda e: self.set_tool_color('gray'))
        self.root.bind('k', lambda e: self.set_tool_color('gray')) # Alt for black
        
        self.root.bind('1', lambda e: self.set_tool_value(1))
        self.root.bind('2', lambda e: self.set_tool_value(2))
        self.root.bind('3', lambda e: self.set_tool_value(3))
        self.root.bind('0', lambda e: self.set_tool_value(0))

        # Initialize Cursor Object (Hidden initially)
        self.cursor_id = self.canvas.create_text(0, 0, text="1", fill="yellow", font=("Arial", 24, "bold"), state='hidden')
        self.cursor_bg_id = self.canvas.create_rectangle(0, 0, 0, 0, fill="black", state='hidden')
        self.canvas.tag_lower(self.cursor_bg_id, self.cursor_id) # Put bg behind text

        self.update_stats_display()
        self.draw_overlays()
        self.update_tool_visuals()

    # --- PAINT TOOL LOGIC ---
    def set_tool_color(self, color):
        self.tool_color = color
        self.update_tool_visuals()

    def set_tool_value(self, val):
        self.tool_value = val
        self.update_tool_visuals()

    def update_tool_visuals(self):
        # Update the info label
        col_hex = self.display_colors[self.tool_color]
        self.info_label.config(text=f"TOOL: {self.tool_color.upper()} [{self.tool_value}]", fg=col_hex)
        
        # Update the cursor object
        self.canvas.itemconfig(self.cursor_id, text=str(self.tool_value), fill=col_hex)

    def on_mouse_move(self, event):
        # Translate window coordinates to canvas coordinates
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        # Move the custom cursor
        self.canvas.coords(self.cursor_id, cx + 15, cy - 15) # Offset slightly to right-up
        
        # Optional: Background for cursor readability
        bbox = self.canvas.bbox(self.cursor_id)
        if bbox:
            self.canvas.coords(self.cursor_bg_id, bbox[0], bbox[1], bbox[2], bbox[3])
        
        self.canvas.itemconfig(self.cursor_id, state='normal')
        self.canvas.itemconfig(self.cursor_bg_id, state='normal')

    def on_mouse_enter(self, event):
        # Hide system cursor when over canvas
        self.canvas.config(cursor="none")
        self.canvas.itemconfig(self.cursor_id, state='normal')
        self.canvas.itemconfig(self.cursor_bg_id, state='normal')

    def on_mouse_leave(self, event):
        # Show system cursor when leaving canvas
        self.canvas.config(cursor="")
        self.canvas.itemconfig(self.cursor_id, state='hidden')
        self.canvas.itemconfig(self.cursor_bg_id, state='hidden')

    def on_left_click(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        if self.setup_mode:
            self.handle_setup_click(canvas_x, canvas_y)
        else:
            # PAINT ACTION
            city = self.find_nearest_city(canvas_x, canvas_y)
            if city:
                # Apply current tool to data
                self.data[city][self.tool_color] = self.tool_value
                # Visual feedback
                self.draw_overlays()
                self.update_stats_display()

    # --- STANDARD LOGIC ---
    def take_burst_screenshots(self):
        paths = []
        base_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting burst capture ({BURST_COUNT} frames)...")
        for i in range(BURST_COUNT):
            filename = f"screen_{base_ts}_{i:02d}.png"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            try:
                pyautogui.screenshot(region=(0, 0, GAME_W, GAME_H)).save(filepath)
                paths.append(filepath)
            except Exception: pass
            if i < BURST_COUNT - 1: time.sleep(BURST_INTERVAL)
        print("Burst done.")
        return paths

    def run_ocr_supply(self):
        """
        Extracts supply regions, applies manual thresholding (White/Gray/Red),
        and runs OCR.
        """
        color_map = {4: "yellow", 5: "gray", 6: "red", 7: "blue"}
        
        for i, (x, y, w, h) in enumerate(OCR_REGIONS):
            if i in color_map:
                try:
                    # Crop from the PIL image directly
                    crop = self.pil_image.crop((x, y, x + w, y + h))
                    
                    # Convert crop to RGB for thresholding logic
                    crop_rgb = crop.convert("RGB")
                    cw, ch = crop_rgb.size
                    pixels = crop_rgb.load()

                    # Manual Thresholding Loop (ported from standalone script)
                    for px in range(cw):
                        for py in range(ch):
                            r, g, b = pixels[px, py]
                            
                            is_white = (r > 200 and g > 200 and b > 200)
                            is_gray = (pixels[px, py] == (162, 163, 163))
                            # Red text fix (High Red, Low Green/Blue)
                            is_red = (r > 130 and g < 100 and b < 100)
                            
                            if is_white or is_gray or is_red:
                                pixels[px, py] = (255, 255, 255)
                            else:
                                pixels[px, py] = (0, 0, 0)

                    # Now process for OCR
                    # Passing crop_rgb because we modified pixels in-place
                    val = try_ocr_best(preprocess_image_ocr(crop_rgb), region_idx=i)
                    
                    if val is not None: 
                        self.ocr_targets[color_map[i]] = 24 - val
                        print(f"OCR Success for {color_map[i]}: Supply={val} -> Board Target={24-val}")
                except Exception as e:
                    print(f"OCR Error for region {i}: {e}")
                    pass

    def run_heuristic(self):
        if not self.cities: return
        hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        box_r = 35 
        global_usage = {"yellow": 0, "red": 0, "blue": 0, "gray": 0}
        self.reset_data()
        
        for name, (cx, cy) in self.cities.items():
            cx, cy = int(cx), int(cy)
            roi = hsv[max(0, cy-box_r):min(GAME_H, cy+box_r), max(0, cx-box_r):min(GAME_W, cx+box_r)]
            if roi.size == 0: continue

            for color_name, (lower, upper) in COLORS_HSV.items():
                target_limit = self.ocr_targets.get(color_name, 24)
                if global_usage[color_name] >= target_limit:
                    self.data[name][color_name] = 0
                    continue

                mask = cv2.inRange(roi, np.array(lower), np.array(upper))
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                count = sum(1 for c in cnts if cv2.contourArea(c) > 30)
                count = min(count, 3)
                final_count = min(count, target_limit - global_usage[color_name])
                
                self.data[name][color_name] = final_count
                global_usage[color_name] += final_count

        self.update_stats_display()
        self.draw_overlays()

    def update_stats_display(self):
        totals = {"yellow": 0, "red": 0, "blue": 0, "gray": 0}
        for city_data in self.data.values():
            for color in totals: totals[color] += city_data[color]
        
        for color in totals:
            current = totals[color]
            target = self.ocr_targets.get(color, 24)
            fg = "#00ff00" if current == target else self.display_colors[color]
            if current > target: fg = "#ff00ff"
            self.stats_labels[color].config(text=f"{color.upper()}: {current} / {target}", fg=fg)

    def _on_mousewheel(self, event): self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    def _on_shift_mousewheel(self, event): self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
    
    def reset_data(self):
        self.data = {}
        for city in self.cities: self.data[city] = {"yellow": 0, "red": 0, "blue": 0, "gray": 0}

    def toggle_setup(self):
        self.setup_mode = not self.setup_mode
        state = "ON" if self.setup_mode else "OFF"
        self.btn_setup.config(text=f"1. Setup Mode ({state})", bg="#ffcccc" if self.setup_mode else "lightgray")
        self.draw_overlays()

    def find_nearest_city(self, x, y, limit=50):
        nearest = None; min_dist = float('inf')
        for name, coords in self.cities.items():
            dist = np.sqrt((coords[0]-x)**2 + (coords[1]-y)**2)
            if dist < limit and dist < min_dist: min_dist = dist; nearest = name
        return nearest

    def on_right_click(self, event):
        if self.setup_mode: return
        cx = self.canvas.canvasx(event.x); cy = self.canvas.canvasy(event.y)
        city = self.find_nearest_city(cx, cy)
        if city:
            self.data[city] = {"yellow": 0, "red": 0, "blue": 0, "gray": 0}
            self.draw_overlays(); self.update_stats_display()

    def handle_setup_click(self, x, y):
        name = simpledialog.askstring("City Setup", "Name:")
        if name:
            self.cities[name] = [x, y]
            if name not in self.data: self.data[name] = {"yellow": 0, "red": 0, "blue": 0, "gray": 0}
            with open(COORDS_FILE, 'w') as f: json.dump(self.cities, f, indent=4)

    def draw_overlays(self):
        # We don't delete cursor IDs, only ui_overlay
        self.canvas.delete("ui_overlay")
        for name, (cx, cy) in self.cities.items():
            col = "red" if self.setup_mode else "white"
            self.canvas.create_oval(cx-5, cy-5, cx+5, cy+5, fill=col, tags="ui_overlay")
            if not self.setup_mode:
                info = [f"{k[0].upper()}:{v}" for k,v in self.data[name].items() if v > 0]
                if info:
                    txt = " ".join(info)
                    self.canvas.create_rectangle(cx-25, cy-25, cx+25, cy-10, fill="black", stipple="gray50", tags="ui_overlay")
                    self.canvas.create_text(cx, cy-18, text=txt, fill="#00ff00", font=("Arial", 10, "bold"), tags="ui_overlay")
            else: self.canvas.create_text(cx, cy-15, text=name, fill="white", tags="ui_overlay")
        
        # Ensure cursor stays on top
        self.canvas.tag_raise(self.cursor_bg_id)
        self.canvas.tag_raise(self.cursor_id)

    def save_dataset(self):
        if not self.burst_paths: return
        current_dataset = []
        if os.path.exists(DATASET_FILE):
            try:
                with open(DATASET_FILE, 'r') as f: current_dataset = json.load(f)
            except: pass
        
        for path in self.burst_paths:
            entry = {"image": path, "annotations": self.data, "ocr_targets": self.ocr_targets}
            found = False
            for i, item in enumerate(current_dataset):
                if item["image"] == path: 
                    current_dataset[i] = entry; found = True; break
            if not found: current_dataset.append(entry)

        with open(DATASET_FILE, 'w') as f: json.dump(current_dataset, f, indent=2)
        if messagebox.askyesno("Saved", "Batch saved! Exit?"): self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    try: app = PandemicLabeler(root); root.mainloop()
    except Exception as e: print(f"Error: {e}")
