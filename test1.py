import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageDraw
import os

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        self.canvas = tk.Canvas(root, bg="white", width=400, height=400)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.save_button = tk.Button(root, text="Save Drawing", command=self.show_save_dialog)
        self.save_button.pack()

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_drawing)
        self.reset_button.pack()

        self.toggle_button = tk.Button(root, text="Toggle Eraser", command=self.toggle_eraser)
        self.toggle_button.pack()

        self.brush_size_label = tk.Label(root, text="Brush Size:")
        self.brush_size_label.pack()
        
        self.brush_size_slider = tk.Scale(root, from_=1, to=10, orient="horizontal")
        self.brush_size_slider.set(2)  # Set the initial brush size
        self.brush_size_slider.pack()

        self.drawing = True  # Start in drawing mode
        self.eraser = False  # Initially, not in eraser mode
        self.last_x = None
        self.last_y = None
        self.brush_size = 2  # Initial brush size
        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            color = "black" if not self.eraser else "white"
            self.canvas.create_line(
                (self.last_x, self.last_y, x, y),
                fill=color,
                width=self.brush_size
            )
            self.draw.line(
                (self.last_x, self.last_y, x, y),
                fill=color,
                width=self.brush_size
            )
            self.last_x = x
            self.last_y = y

    def toggle_eraser(self):
        self.eraser = not self.eraser

    def show_save_dialog(self):
        folder_name = simpledialog.askstring("Save Drawing", "What is this?")
        if folder_name:
            folder_path = os.path.join("Drawing Library", folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            file_name = folder_name + "1"
            if file_name:
                file_path = os.path.join(folder_path, file_name + ".png")
                self.image.save(file_path)

    def reset_drawing(self):
        self.canvas.delete("all")  # Clear the canvas
        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)

    def update_brush_size(self, value):
        self.brush_size = int(value)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    app.brush_size_slider.config(command=app.update_brush_size)
    root.mainloop()
