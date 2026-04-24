import threading
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox, font
import scipy.signal as signal
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# MindRove / BrainFlow Imports
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

# ================= 1. MODEL DEFINITION =================
class FinalPushLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.4, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        self.lstm.flatten_parameters()   
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])    

# ================= 2. UTILITIES & FILTERING =================
def filter_emg_data(data, fs=1000.0):
    """Notch-60 Hz + 10-200 Hz bandpass, channel-by-channel via SOS."""
    if data.shape[0] < 27:
        return data.copy().astype(float)
    notch_freq, Q = 60.0, 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)
    sos_notch = signal.tf2sos(b_notch, a_notch)
    sos_band  = signal.butter(4, [10.0, 200.0], btype='band', fs=fs, output='sos')

    filtered = np.zeros_like(data, dtype=float)
    for ch in range(data.shape[1]):
        ch_data = signal.sosfiltfilt(sos_notch, data[:, ch])
        ch_data = signal.sosfiltfilt(sos_band,  ch_data)
        filtered[:, ch] = ch_data
    return filtered

def apply_filters(data, fs=500.0):
    return filter_emg_data(data, fs=fs)

# ================= 2b. WINDOWING =================
def create_sequences(df, window_size, step):
    """Create (X, y) tensors from a DataFrame with CH1-CH4 + Class columns.
    Only windows where every sample shares the same label are kept."""
    if df.empty:
        return torch.tensor([]), torch.tensor([])

    df = df.dropna(subset=['Class'])
    channels = ['CH1', 'CH2', 'CH3', 'CH4']
    data   = df[channels].values
    labels = df['Class'].values.astype(int)

    X, y = [], []
    for i in range(0, len(data) - window_size, step):
        window_labels = labels[i:i + window_size]
        if np.all(window_labels == window_labels[0]):
            X.append(data[i:i + window_size])
            y.append(window_labels[0])

    if not X:
        return torch.tensor([]), torch.tensor([])

    return (torch.tensor(np.stack(X), dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))

# ================= 2c. LIVE DATA LOADER (replaces load_all_subjects) =================
def load_mindrove_data(filepath, fs=500.0):
    """Load, filter, and normalize a MindRove CSV recording.

    The CSV written by _sequence_worker has integer columns (0,1,2,3) + 'label'.
    Returns (df with CH1-CH4 + label columns, fitted StandardScaler).
    """
    df = pd.read_csv(filepath)
    # Support both named CH1-CH4 columns (new _sequence_worker) and
    # legacy integer column names (0,1,2,3) from older recordings.
    if 'CH1' in df.columns:
        emg_cols = ['CH1', 'CH2', 'CH3', 'CH4']
    else:
        raw_emg_cols = [c for c in df.columns if c != 'label'][:4]
        rename_map   = {raw_emg_cols[i]: f'CH{i+1}' for i in range(len(raw_emg_cols))}
        df = df.rename(columns=rename_map)
        emg_cols = ['CH1', 'CH2', 'CH3', 'CH4']
    label_vals = df['label'].values
    df = df[emg_cols].copy()
    df['label'] = label_vals

    channels = ['CH1', 'CH2', 'CH3', 'CH4']

    # Step 1 – filter
    df[channels] = filter_emg_data(df[channels].values, fs=fs)

    # Step 2 – normalize (StandardScaler, matching notebook's per-subject scaling)
    scaler = StandardScaler()
    df[channels] = scaler.fit_transform(df[channels].values)

    return df, scaler

# ================= 3. CONFIGURATION =================
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD
NUM_CHANNELS = 4
SAMPLING_RATE = 500
WINDOW_SIZE = 50
STEP_SIZE = 10
MODEL_PATH = "bilstm_vader.pt"
DATA_FILE = "collected_emg_data.csv"

PLOT_SAMPLES = 250

GUIDED_SEQUENCE = [
    {"label": 0, "name": "RELAX",  "duration": 10},
    {"label": 1, "name": "FIST",   "duration": 10},
    {"label": 2, "name": "ROCK",   "duration": 10},
    {"label": 3, "name": "PEACE",  "duration": 10},
    {"label": 4, "name": "SHAKA",  "duration": 10}
]

THEMES = {
    'Dark': {
        'bg': '#121212', 'fg': '#00FF00', 'text_fg': 'white',
        'btn_bg': '#FF9800', 'plot_bg': '#121212', 'plot_fg': 'white',
        'emg_color': 'cyan', 'accent': '#2196F3'
    },
    'Light': {
        'bg': '#f0f0f0', 'fg': '#8b0000', 'text_fg': '#111111',
        'btn_bg': '#c0392b', 'plot_bg': '#ffffff', 'plot_fg': '#111111',
        'emg_color': '#0055cc', 'accent': '#007700'
    },
    'High Contrast': {
        'bg': 'black', 'fg': 'yellow', 'text_fg': 'yellow',
        'btn_bg': '#555500', 'plot_bg': 'black', 'plot_fg': 'yellow',
        'emg_color': 'white', 'accent': 'yellow'
    },
}

CHANNEL_COLORS = ['cyan', '#FF9800', '#1DB954', '#f44336']


# ================= 4. GUI CLASS =================
class GuidedBiLSTM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MindRove Pro: Collect & Predict")

        self.current_theme = 'Dark'
        self.title_font = font.Font(family="Helvetica", size=40)
        self.offset_step = 200

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FinalPushLSTM(input_size=NUM_CHANNELS, num_classes=5).to(self.device)
        self._load_model()

        self.board = None
        self.emg_channels = BoardShim.get_emg_channels(BOARD_ID)

        self.is_running = False
        self.is_predicting = False
        self.last_stable_prediction = 0

        self.plot_buffer = np.zeros((PLOT_SAMPLES, NUM_CHANNELS))
        self.live_window_buffer = np.zeros((0, NUM_CHANNELS), dtype=np.float32)

        self._setup_ui()
        self._init_plots()
        self._apply_theme('Dark')
        self._check_device_connection()

    # ------------------------------------------------------------------
    def _setup_ui(self):
        self.conn_lbl = tk.Label(self.root, text="DEVICE: DISCONNECTED",
                                 font=("Courier", 12, "bold"))
        self.conn_lbl.pack(pady=5)

        self.status_lbl = tk.Label(self.root, text="System Ready",
                                   font=("Courier", 18, "bold"))
        self.status_lbl.pack(pady=10)

        self.timer_lbl = tk.Label(self.root, text="00.0s", font=self.title_font)
        self.timer_lbl.pack()

        self.acc_frame = tk.Frame(self.root)
        self.acc_frame.pack(pady=5)
        self.train_acc_lbl = tk.Label(self.acc_frame, text="Train Acc: --%")
        self.train_acc_lbl.pack(side="left", padx=10)
        self.test_acc_lbl = tk.Label(self.acc_frame, text="Test Acc: --%")
        self.test_acc_lbl.pack(side="left", padx=10)

        self.check_frame = tk.Frame(self.root)
        self.check_frame.pack(pady=5)
        self.checks = {}
        for item in GUIDED_SEQUENCE:
            lbl = tk.Label(self.check_frame, text=f"[ ] {item['name']}")
            lbl.pack(anchor="w")
            self.checks[item['label']] = lbl

        self.progress = ttk.Progressbar(self.root, length=400)
        self.progress.pack(pady=10)

        self.ctrl_frame = tk.Frame(self.root)
        self.ctrl_frame.pack(pady=10)
        self.btn_conn  = tk.Button(self.ctrl_frame, text="CONNECT DEVICE",
                                   command=self._connect_board)
        self.btn_conn.pack(side="left", padx=5)
        self.btn_seq   = tk.Button(self.ctrl_frame, text="START SEQUENCE",
                                   command=self.run_sequence)
        self.btn_seq.pack(side="left", padx=5)
        self.btn_train = tk.Button(self.ctrl_frame, text="TRAIN",
                                   command=self.start_training)
        self.btn_train.pack(side="left", padx=5)
        self.btn_live  = tk.Button(self.ctrl_frame, text="LIVE PREDICT",
                                   command=self.toggle_prediction)
        self.btn_live.pack(side="left", padx=5)
        tk.Button(self.ctrl_frame, text="Accessibility",
                  command=self._open_accessibility,
                  bg='gray', fg='white').pack(side='left', padx=5)

    # ------------------------------------------------------------------
    def _open_accessibility(self):
        win = tk.Toplevel(self.root)
        win.title("Accessibility")
        t = THEMES[self.current_theme]
        win.configure(bg=t['bg'])
        tk.Label(win, text="Display font size",
                 bg=t['bg'], fg=t['text_fg']).pack(pady=5)
        tk.Scale(win, from_=14, to=70, orient='horizontal',
                 bg=t['bg'], fg=t['text_fg'],
                 command=lambda v: self.title_font.configure(size=int(v))).pack()
        tk.Label(win, text="Color theme",
                 bg=t['bg'], fg=t['text_fg']).pack(pady=5)
        for name in THEMES:
            tk.Radiobutton(win, text=name, value=name,
                           bg=t['bg'], fg=t['text_fg'],
                           command=lambda n=name: self._apply_theme(n)).pack()
        tk.Button(win, text="Close", command=win.destroy).pack(pady=10)

    # ------------------------------------------------------------------
    def _apply_theme(self, name):
        self.current_theme = name
        t = THEMES[name]
        fg = t['plot_fg']

        self.root.configure(bg=t['bg'])
        self.acc_frame.configure(bg=t['bg'])
        self.check_frame.configure(bg=t['bg'])
        self.ctrl_frame.configure(bg=t['bg'])

        for lbl in [self.conn_lbl, self.status_lbl, self.timer_lbl,
                    self.train_acc_lbl, self.test_acc_lbl]:
            lbl.configure(bg=t['bg'], fg=t['text_fg'])
        for lbl in self.checks.values():
            lbl.configure(bg=t['bg'], fg=t['text_fg'])
        for btn in [self.btn_conn, self.btn_seq, self.btn_train, self.btn_live]:
            btn.configure(bg=t['btn_bg'], fg='white')

        self.ax.set_facecolor(t['plot_bg'])
        self.fig.patch.set_facecolor(t['plot_bg'])

        for spine in self.ax.spines.values():
            spine.set_edgecolor(fg)

        self.ax.tick_params(colors=fg, labelcolor=fg, which='both')
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        self.ax.title.set_color(fg)

        self._redraw_legend(fg)
        self.canvas.draw_idle()

    def _redraw_legend(self, fg):
        legend = self.ax.get_legend()
        if legend:
            legend.remove()
        handles = [
            Line2D([0], [0], color=CHANNEL_COLORS[i], lw=1.5, label=f'CH{i+1}')
            for i in range(NUM_CHANNELS)
        ]
        leg = self.ax.legend(handles=handles, loc='upper right',
                             fontsize=7, framealpha=0.3)
        for text in leg.get_texts():
            text.set_color(fg)

    # ------------------------------------------------------------------
    def _init_plots(self):
        self.fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax  = self.fig.add_subplot(111)

        self.ax.set_xlim(0, PLOT_SAMPLES - 1)
        self.ax.set_ylim(-self.offset_step,
                         NUM_CHANNELS * self.offset_step + self.offset_step)
        self.ax.set_xlabel("Samples", fontsize=8)
        self.ax.set_ylabel("Amplitude (μV)", fontsize=8)
        self.ax.set_title("Live EMG", fontsize=9)

        x = np.arange(PLOT_SAMPLES)
        self.lines = [
            self.ax.plot(x,
                         np.zeros(PLOT_SAMPLES) + i * self.offset_step,
                         lw=1, color=CHANNEL_COLORS[i])[0]
            for i in range(NUM_CHANNELS)
        ]

        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        threading.Thread(target=self._update_plot_loop, daemon=True).start()

    def _update_plot_loop(self):
        while True:
            if self.board and self.board.is_prepared():
                try:
                    chunk = self.board.get_current_board_data(PLOT_SAMPLES)
                    new_emg = chunk[self.emg_channels, :].T
                    new_emg = new_emg[:, :NUM_CHANNELS]

                    n = len(new_emg)
                    if n >= PLOT_SAMPLES:
                        self.plot_buffer = new_emg[-PLOT_SAMPLES:]
                    else:
                        self.plot_buffer = np.roll(self.plot_buffer, -n, axis=0)
                        self.plot_buffer[-n:] = new_emg

                    display = (apply_filters(self.plot_buffer)
                               if self.plot_buffer.shape[0] >= 15
                               else self.plot_buffer)

                    for i, line in enumerate(self.lines):
                        line.set_ydata(display[:, i] + i * self.offset_step)

                    self.root.after(0, self.canvas.draw_idle)
                except Exception as e:
                    print(f"Plot error: {e}")

            time.sleep(0.05)

    # ------------------------------------------------------------------
    def _check_device_connection(self):
        if self.board and self.board.is_prepared():
            self.conn_lbl.config(text="DEVICE: CONNECTED", fg="#1DB954")
        else:
            self.conn_lbl.config(text="DEVICE: DISCONNECTED", fg="red")
        self.root.after(2000, self._check_device_connection)

    def _connect_board(self):
        params = MindRoveInputParams()
        try:
            self.board = BoardShim(BOARD_ID, params)
            self.board.prepare_session()
            self.board.start_stream()
            messagebox.showinfo("Success", "MindRove Connected!")
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    # ------------------------------------------------------------------
    def run_sequence(self):
        if not self.board:
            messagebox.showwarning("Error", "Connect device first!")
            return
        if self.is_running:
            return
        self.is_running = True
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        threading.Thread(target=self._sequence_worker, daemon=True).start()

    def _sequence_worker(self):
        for item in GUIDED_SEQUENCE:
            label, name, duration = item['label'], item['name'], item['duration']
            self.root.after(0, lambda n=name: self.status_lbl.config(text=f"PERFORM: {n}"))
            start_time = time.time()
            while time.time() - start_time < duration:
                rem = max(0, duration - (time.time() - start_time))
                self.root.after(0, lambda r=rem: self.timer_lbl.config(text=f"{r:.1f}s"))
                waited = 0
                while self.board.get_board_data_count() < STEP_SIZE and waited < 50:
                    time.sleep(0.002)
                    waited += 1
                data = self.board.get_board_data(STEP_SIZE)
                emg_data = data[self.emg_channels, :].T
                emg_data = emg_data[:, :NUM_CHANNELS]   
                if emg_data.size > 0:
                    df = pd.DataFrame(emg_data, columns=['CH1', 'CH2', 'CH3', 'CH4'])
                    df['label'] = label
                    write_header = not os.path.exists(DATA_FILE)
                    df.to_csv(DATA_FILE, mode='a', header=write_header, index=False)
                time.sleep(0.02)
            self.root.after(0, lambda l=label: self.checks[l].config(
                text=f"[X] {GUIDED_SEQUENCE[l]['name']}", fg="#1DB954"))
        self.is_running = False
        self.root.after(0, lambda: self.status_lbl.config(text="Recording Complete"))

    # ------------------------------------------------------------------
    def start_training(self):
        self.status_lbl.config(text="Training...")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        if not os.path.exists(DATA_FILE):
            self.root.after(0, lambda: self.status_lbl.config(
                text="ERROR: No data file found. Run sequence first."))
            return

        self.model = FinalPushLSTM(input_size=NUM_CHANNELS, num_classes=5).to(self.device)

        # ── Step 1 & 2: filter_emg_data + StandardScaler via load_mindrove_data
        df, self.scaler = load_mindrove_data(DATA_FILE, fs=float(SAMPLING_RATE))

        # ── Step 3: windowing via create_sequences (strict label-consistency)
        df = df.rename(columns={'label': 'Class'})
        X, y = create_sequences(df, window_size=WINDOW_SIZE, step=STEP_SIZE)

        if X.shape[0] == 0:
            self.root.after(0, lambda: self.status_lbl.config(
                text="ERROR: No valid windows — re-record data."))
            return

        X_np, y_np = X.numpy(), y.numpy()
        # stratify=y_np ensures each class is proportionally represented in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test,  dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test  = torch.tensor(y_test,  dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=32, shuffle=True)


        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        epochs = 15
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.003,
            steps_per_epoch=len(train_loader), epochs=epochs)

        # ── Training loop ─────────────────────────────────────────────────────
        self.model.train()
        for epoch in range(epochs):
            correct, total = 0, 0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out  = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                scheduler.step()          # OneCycleLR steps per batch
                correct += (out.argmax(1) == by).sum().item()
                total   += by.size(0)
            acc = (correct / total) * 100
            self.root.after(0,
                lambda a=acc: self.train_acc_lbl.config(text=f"Train Acc: {a:.1f}%"))
            self.progress['value'] = (epoch + 1) * (100 / epochs)

        # ── Evaluation ────────────────────────────────────────────────────────
        self.model.eval()
        with torch.no_grad():
            test_out = self.model(X_test.to(self.device))
            test_acc = (test_out.argmax(1) == y_test.to(self.device)
                        ).float().mean().item() * 100

        torch.save(self.model.state_dict(), MODEL_PATH)
        # Persist scaler params so live prediction uses identical normalisation
        np.save("scaler_mean.npy",  self.scaler.mean_)
        np.save("scaler_scale.npy", self.scaler.scale_)

        self.root.after(0, lambda: [
            self.test_acc_lbl.config(text=f"Test Acc: {test_acc:.1f}%"),
            self.status_lbl.config(text="Training Done!")
        ])

    # ------------------------------------------------------------------
    def toggle_prediction(self):
        if not self.board:
            messagebox.showwarning("Error", "Connect device first!")
            return
        self.is_predicting = not self.is_predicting
        if self.is_predicting:
            self.btn_live.config(text="STOP PREDICT", bg="#f44336")
            self.status_lbl.config(text="LIVE PREDICTING")
            threading.Thread(target=self._live_loop, daemon=True).start()
        else:
            self.btn_live.config(text="LIVE PREDICT",
                                 bg=THEMES[self.current_theme]['btn_bg'])
            self.status_lbl.config(text="Predicting Stopped")
            self.timer_lbl.config(text="00.0s")

    def _update_prediction_ui(self, gesture_name, confidence):
        self.timer_lbl.config(text=gesture_name)
        self.status_lbl.config(
            text=f"LIVE PREDICTING  |  {gesture_name}  |  Conf: {confidence:.0f}%"
        )

    def _read_fresh_samples(self, target_samples):
        """Read exactly target_samples new rows from the board buffer."""
        collected = []
        total_samples = 0

        while total_samples < target_samples:
            if not self.is_predicting:
                return None

            available = self.board.get_board_data_count()
            if available < 1:
                time.sleep(0.001)
                continue

            samples_to_read = min(available, target_samples - total_samples)
            data = self.board.get_board_data(samples_to_read)
            if data is None or data.shape[1] == 0:
                continue

            chunk = data[self.emg_channels, :].T[:, :NUM_CHANNELS]
            if chunk.size == 0:
                continue

            collected.append(chunk)
            total_samples += chunk.shape[0]

        return np.vstack(collected)[:target_samples]

    def _record_window(self, scaler_mean, scaler_scale):
        needed = WINDOW_SIZE if self.live_window_buffer.shape[0] < WINDOW_SIZE else STEP_SIZE
        fresh = self._read_fresh_samples(needed)
        if fresh is None:
            return None

        if self.live_window_buffer.size == 0:
            self.live_window_buffer = fresh
        else:
            self.live_window_buffer = np.vstack((self.live_window_buffer, fresh))

        if self.live_window_buffer.shape[0] > WINDOW_SIZE:
            self.live_window_buffer = self.live_window_buffer[-WINDOW_SIZE:]

        if self.live_window_buffer.shape[0] < WINDOW_SIZE:
            return None

        raw = self.live_window_buffer.astype(np.float32, copy=True)
        filtered = filter_emg_data(raw, fs=float(SAMPLING_RATE))
        normalized = (filtered - scaler_mean) / scaler_scale
        return normalized.astype(np.float32)

    def _live_loop(self):
        self.board.get_board_data()
        self.live_window_buffer = np.zeros((0, NUM_CHANNELS), dtype=np.float32)

        # ── Load normalisation stats saved during training ────────────────────
        if hasattr(self, 'scaler'):
            scaler_mean  = self.scaler.mean_
            scaler_scale = self.scaler.scale_
        elif (os.path.exists("scaler_mean.npy") and
              os.path.exists("scaler_scale.npy")):
            scaler_mean  = np.load("scaler_mean.npy")
            scaler_scale = np.load("scaler_scale.npy")
        else:
            self.root.after(0, lambda: self.status_lbl.config(
                text="ERROR: Train model first — no normalisation stats found"))
            self.is_predicting = False
            self.root.after(0, lambda: self.btn_live.config(
                text="LIVE PREDICT",
                bg=THEMES[self.current_theme]['btn_bg']))
            return

        self.model.eval()
        VOTES = 10  

        while self.is_predicting:
            try:
                votes = np.zeros(len(GUIDED_SEQUENCE), dtype=int)
                prob_sum = np.zeros(len(GUIDED_SEQUENCE), dtype=np.float32)
                windows_used = 0

                for _ in range(VOTES):
                    if not self.is_predicting:
                        break

                    window = self._record_window(scaler_mean, scaler_scale)
                    if window is None:
                        break

                    inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.model(inp)
                        probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
                        pred = out.argmax(1).item()

                    if 0 <= pred < len(GUIDED_SEQUENCE):
                        votes[pred] += 1
                        prob_sum += probs
                        windows_used += 1

                if not self.is_predicting:
                    break

                if windows_used == 0:
                    continue

                avg_probs = prob_sum / windows_used
                vote_winner = int(votes.argmax())
                prob_winner = int(avg_probs.argmax())
                cls = prob_winner if votes[prob_winner] == votes[vote_winner] else vote_winner
                confidence = max(avg_probs[cls] * 100, (votes[cls] / windows_used) * 100)
                gesture_name = GUIDED_SEQUENCE[cls]['name']

                print(
                    f"votes={votes} probs={np.round(avg_probs, 3)} "
                    f"→ {gesture_name} ({confidence:.0f}%)"
                )

                self.last_stable_prediction = cls
                self.root.after(0, lambda n=gesture_name, c=confidence:
                    self._update_prediction_ui(n, c))

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Prediction error: {e}")

    # ------------------------------------------------------------------
    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(
                    torch.load(MODEL_PATH, map_location=self.device))
            except Exception:
                pass
        if (os.path.exists("scaler_mean.npy") and
                os.path.exists("scaler_scale.npy")):
            from sklearn.preprocessing import StandardScaler as _SS
            sc = _SS()
            sc.mean_  = np.load("scaler_mean.npy")
            sc.scale_ = np.load("scaler_scale.npy")
            sc.var_   = sc.scale_ ** 2
            sc.n_features_in_ = NUM_CHANNELS
            self.scaler = sc


if __name__ == '__main__':
    root = tk.Tk()
    app = GuidedBiLSTM_GUI(root)
    root.mainloop()
