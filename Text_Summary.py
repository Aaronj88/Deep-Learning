import tkinter as tk
from testing import predict_statement


def close_window():
    text = text_box.get()
    print("TEST TEXT HERE")
    print(text)
    print("TEST TEXT HERE")
    pred = predict_statement(text)
    print(pred)
    window.destroy()

text = ""


window = tk.Tk()
window.geometry("800x800")
window.title("Text Analysis")
window.config(bg="SkyBlue4")


label = tk.Label(window,text = "Enter Your Text",font = ("Times",35),fg = "CadetBlue",bg = "SkyBlue4")
label.pack(pady = 10)


text_box = tk.Entry(window,font = ("Times",20),bg = "RoyalBlue4",fg = "CadetBlue")
text_box.pack(pady = 5)


submit_button = tk.Button(window,text="Submit",font = ("Times",20),width = 20,height=1,command = close_window,bg = "RoyalBlue4",fg = "CadetBlue")
submit_button.pack(pady = 10)


window.mainloop()