import tkinter as tk
from tkinter import filedialog


def select_file(label: tk.Label) -> None:
    """Open a file dialog to select an MP3 file."""
    filetypes = [("MP3 files", "*.mp3"), ("All files", "*.*")]
    filename = filedialog.askopenfilename(title="Select an MP3 file", filetypes=filetypes)
    if filename:
        label.config(text=filename)


def main() -> None:
    root = tk.Tk()
    root.title("MP3 Uploader")
    root.geometry("400x150")

    label = tk.Label(root, text="No file selected")
    label.pack(pady=10)

    button = tk.Button(root, text="Upload MP3", command=lambda: select_file(label))
    button.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    main()
