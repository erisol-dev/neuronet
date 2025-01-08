import tkinter as tk
from tkinter import ttk, Canvas
from turtle import RawTurtle, TurtleScreen
from tkinter import colorchooser

class App(ttk.Frame):
  def __init__(self, master):
    super().__init__(master)
    self.pack(fill="both", expand=True)
    self.master = master
    self.master.update()

    #brush
    self.brush_size = tk.DoubleVar(value=10)
    self.brush_color = tk.StringVar(value="#000000")

    self.brush_values = [_ for _ in range(5, 31)]

    #create canvas
    self.create_canvas()

    #create buttons
    self.create_buttons()

    #create output display
    self.create_display()

  def create_canvas(self):
    #canvas frame
    self.canvas = Canvas(self)
    self.canvas.place(x=0, y=0, width=self.master.winfo_width(), height=self.master.winfo_height()*0.8)
    self.canvas.configure(highlightthickness=0)

    # Create a TurtleScreen from the Canvas
    self.turtle_screen = TurtleScreen(self.canvas)
    self.turtle_screen.bgcolor("red")
  
  def create_buttons(self):
    style = ttk.Style()
    style.configure("BG.TFrame", background="white")
    buttons_frame = ttk.Frame(self, style="BG.TFrame", width=self.master.winfo_width()/2, height=self.master.winfo_height()*0.2, borderwidth=5, relief="solid")
    buttons_frame.place(x=0, y=self.master.winfo_height()*0.8)

    #stops tkinter frame from changing to fit children
    buttons_frame.grid_propagate(False)

    # Get the number of rows and columns dynamically (example for 3x3 grid)
    total_rows = 2
    total_columns = 2

    # Dynamically configure all rows and columns
    for row in range(total_rows):
        buttons_frame.grid_rowconfigure(row, weight=1)

    for col in range(total_columns):
        buttons_frame.grid_columnconfigure(col, weight=1)


    #create brush size picker
    # Create a label for the combobox
    label = ttk.Label(buttons_frame, text="Brush Size", background="white")
    label.grid(row=0, column=0, sticky="e",  padx=10)


    # Create the combobox
    dropdown = ttk.Scale(buttons_frame, from_=0, to=100, orient="horizontal", variable=self.brush_size)
    dropdown.grid(row=0, column=1, sticky="w")

    #create color pick button
    color_button = ttk.Button(buttons_frame, text=f"Color: {self.brush_color.get()}", textvariable=self.brush_color, command=self.choose_color)
    color_button.grid(row=1, column=0, columnspan=2,)

  def create_display(self):
    pass

  def choose_color(self):
    color = colorchooser.askcolor(title ="Choose color") 
    if color:
      self.brush_color = color
      self.color_button.config(text=f"Color: {self.brush_color.get()}")




def init_window(width=800, height=800):
  root = tk.Tk()
  root.geometry(f"{width}x{height}")
  root.resizable(False, False)
  root.title("Model Evaluation for MNIST")
  return root


def main():
  window = init_window()

  main_app = App(window)
  main_app.mainloop()


if __name__ == "__main__":
  main()