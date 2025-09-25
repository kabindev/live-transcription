import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, messagebox
import os

class TranscriptionUI:
    def __init__(self, root, start_callback, stop_callback, send_to_ai_callback, set_api_key_callback):
        self.root = root
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.send_to_ai_callback = send_to_ai_callback
        self.set_api_key_callback = set_api_key_callback

        self.root.title("Phoenix Scribe - AI Transcription")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")

        self.speaker_colors = {
            'SPEAKER_00': '#1abc9c', 'SPEAKER_01': '#3498db',
            'SPEAKER_02': '#9b59b6', 'SPEAKER_03': '#f1c40f',
            'SPEAKER_04': '#e67e22', 'SPEAKER_05': '#e74c3c'
        }

        self.setup_styles()
        self.create_main_widgets()
        self.create_overlay_window()
        self.create_menu()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#2c3e50")
        style.configure("TButton", background="#3498db", foreground="white", font=('Helvetica', 10, 'bold'))
        style.map("TButton", background=[('active', '#2980b9')])
        style.configure("TLabel", background="#2c3e50", foreground="white", font=('Helvetica', 10))
        style.configure("Header.TLabel", font=('Helvetica', 14, 'bold'))
        style.configure("Status.TLabel", font=('Helvetica', 9))

    def create_main_widgets(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side: Transcript and Controls
        left_frame = ttk.Frame(main_pane, padding=10)
        self.create_left_panel(left_frame)
        main_pane.add(left_frame, weight=2)

        # Right side: AI Assistant
        right_frame = ttk.Frame(main_pane, padding=10)
        self.create_right_panel(right_frame)
        main_pane.add(right_frame, weight=1)

    def create_left_panel(self, parent):
        # Controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = ttk.Button(controls_frame, text="Start Recording", command=self.start_callback)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn = ttk.Button(controls_frame, text="Stop Recording", command=self.stop_callback, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Device Selection
        self.device_var = tk.StringVar()
        self.device_menu = ttk.Combobox(controls_frame, textvariable=self.device_var, state="readonly", width=40)
        self.device_menu.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Search and Transcript Area
        transcript_frame = ttk.LabelFrame(parent, text="Full Transcript", padding=10)
        transcript_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        search_frame = ttk.Frame(transcript_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))

        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.search_entry.bind("<Return>", self.search_transcript)

        self.search_button = ttk.Button(search_frame, text="Search", command=self.search_transcript)
        self.search_button.pack(side=tk.LEFT)

        self.transcript_text = scrolledtext.ScrolledText(transcript_frame, wrap=tk.WORD, bg="#34495e", fg="white", insertbackground="white")
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
        self.configure_tags()

        # Status Bar
        self.status_label = ttk.Label(parent, text="Ready", style="Status.TLabel")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def create_right_panel(self, parent):
        ai_frame = ttk.LabelFrame(parent, text="AI Assistant", padding=10)
        ai_frame.pack(fill=tk.BOTH, expand=True)

        # AI Chat History
        self.ai_chat_history = scrolledtext.ScrolledText(ai_frame, wrap=tk.WORD, bg="#34495e", fg="white", state=tk.DISABLED)
        self.ai_chat_history.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.ai_chat_history.tag_configure("user", foreground="#1abc9c", font=('Helvetica', 10, 'bold'))
        self.ai_chat_history.tag_configure("assistant", foreground="#ffffff")
        self.ai_chat_history.tag_configure("error", foreground="#e74c3c")

        # AI Input
        input_frame = ttk.Frame(ai_frame)
        input_frame.pack(fill=tk.X)

        self.ai_input = ttk.Entry(input_frame, font=('Helvetica', 10))
        self.ai_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.ai_input.bind("<Return>", lambda event: self.send_ai_query())

        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_ai_query)
        self.send_btn.pack(side=tk.LEFT)

    def create_overlay_window(self):
        self.overlay = tk.Toplevel(self.root)
        self.overlay.geometry("600x80+50+50")
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.75)
        self.overlay.configure(bg='#2c3e50')

        self.overlay_label = tk.Label(self.overlay, text="", font=('Helvetica', 14, 'bold'), fg='white', bg='#2c3e50', wraplength=580)
        self.overlay_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Make it draggable
        self.overlay.bind("<Button-1>", self.start_move)
        self.overlay.bind("<ButtonRelease-1>", self.stop_move)
        self.overlay.bind("<B1-Motion>", self.do_move)
        self.overlay_x = 0
        self.overlay_y = 0

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Transcript", command=self.export_transcript)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Settings Menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Set Gemini API Key", command=self.prompt_for_gemini_key)
        settings_menu.add_command(label="Set HuggingFace Token", command=self.prompt_for_hf_token)

    def configure_tags(self):
        for speaker, color in self.speaker_colors.items():
            self.transcript_text.tag_configure(speaker, foreground=color, font=('Helvetica', 10, 'bold'))
        self.transcript_text.tag_configure("event", foreground="#f39c12", font=('Helvetica', 10, 'italic'))
        self.transcript_text.tag_configure("timestamp", foreground="#bdc3c7", font=('Helvetica', 9))
        self.transcript_text.tag_configure("search_highlight", background="yellow", foreground="black")

    def search_transcript(self, event=None):
        """Highlights search term occurrences in the transcript."""
        search_term = self.search_var.get()
        self.transcript_text.tag_remove("search_highlight", "1.0", tk.END)

        if not search_term:
            return

        start_pos = "1.0"
        while True:
            start_pos = self.transcript_text.search(search_term, start_pos, stopindex=tk.END, nocase=True)
            if not start_pos:
                break
            end_pos = f"{start_pos}+{len(search_term)}c"
            self.transcript_text.tag_add("search_highlight", start_pos, end_pos)
            start_pos = end_pos

    def update_transcript(self, data):
        timestamp = data['timestamp'].strftime('%H:%M:%S')
        speaker = data['speaker']
        text = data['text']
        events = data['events']

        # Insert timestamp
        self.transcript_text.insert(tk.END, f"[{timestamp}] ", "timestamp")

        if text:
            # Insert speaker and text
            self.transcript_text.insert(tk.END, f"{speaker}: ", speaker)
            self.transcript_text.insert(tk.END, f"{text}\n")

        if events:
            event_str = ", ".join([f"{e['emoji']} {e['event']}" for e in events])
            self.transcript_text.insert(tk.END, f"Events: {event_str}\n", "event")

        self.transcript_text.see(tk.END)

    def update_overlay(self, text):
        self.overlay_label.config(text=text)

    def send_ai_query(self):
        query = self.ai_input.get()
        if not query:
            return

        self.ai_input.delete(0, tk.END)
        self.add_ai_message(f"You: {query}\n", "user")
        self.send_to_ai_callback(query)

    def add_ai_message(self, message, tag):
        self.ai_chat_history.config(state=tk.NORMAL)
        self.ai_chat_history.insert(tk.END, message, tag)
        self.ai_chat_history.config(state=tk.DISABLED)
        self.ai_chat_history.see(tk.END)

    def add_ai_response(self, response):
        self.add_ai_message(f"Assistant: {response}\n", "assistant")

    def update_ai_response(self, response):
        # This is a bit of a hack to replace the "Thinking..." message
        self.ai_chat_history.config(state=tk.NORMAL)
        # Find the start of the last line
        last_line_start = self.ai_chat_history.index("end-2l")
        # Delete the last line
        self.ai_chat_history.delete(last_line_start, tk.END)
        self.ai_chat_history.config(state=tk.DISABLED)
        # Add the final response
        self.add_ai_response(response)

    def set_recording_state(self, is_recording):
        if is_recording:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def update_status(self, message, color):
        self.status_label.config(text=message, foreground=color)

    def populate_device_list(self, devices):
        self.device_menu['values'] = [f"{dev['id']}: {dev['name']}" for dev in devices]
        if devices:
            self.device_menu.set(f"{devices[0]['id']}: {devices[0]['name']}")

    def get_selected_device_id(self):
        try:
            return int(self.device_var.get().split(':')[0])
        except:
            return None # Default device

    def get_full_transcript(self):
        return self.transcript_text.get(1.0, tk.END)

    def clear_transcript(self):
        self.transcript_text.delete(1.0, tk.END)

    def export_transcript(self):
        content = self.get_full_transcript()
        if not content.strip():
            messagebox.showinfo("Export", "Nothing to export.")
            return

        filepath = tk.filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Success", f"Transcript saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")

    def prompt_for_gemini_key(self):
        key = simpledialog.askstring("API Key", "Enter your Google Gemini API Key:", show='*')
        if key:
            self.set_api_key_callback(key)
            self.add_ai_response("Gemini API Key has been set.")

    def show_huggingface_token_dialog(self):
        messagebox.showinfo(
            "Hugging Face Token Required",
            "Speaker diarization requires a Hugging Face authentication token. "
            "Please set the HUGGINGFACE_TOKEN environment variable or enter it in Settings -> Set HuggingFace Token."
        )

    def prompt_for_hf_token(self):
        token = simpledialog.askstring("Hugging Face Token", "Enter your Hugging Face Token:", show='*')
        if token:
            os.environ["HUGGINGFACE_TOKEN"] = token
            messagebox.showinfo("Success", "Hugging Face token set for this session. Restart may be needed to load diarization model.")


    # --- Overlay Dragging Methods ---
    def start_move(self, event):
        self.overlay_x = event.x
        self.overlay_y = event.y

    def stop_move(self, event):
        self.overlay_x = None
        self.overlay_y = None

    def do_move(self, event):
        deltax = event.x - self.overlay_x
        deltay = event.y - self.overlay_y
        x = self.overlay.winfo_x() + deltax
        y = self.overlay.winfo_y() + deltay
        self.overlay.geometry(f"+{x}+{y}")