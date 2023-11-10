import tkinter as tk
from tkinter import filedialog, messagebox

from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image

class ImageToIconConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Image to Icon Converter")
        self.file_path = None

        self.label = tk.Label(root, text="拖动 .jpg 文件到这里")
        self.label.pack(padx=20, pady=20)

        self.convert_button = tk.Button(root, text="转.ico", command=self.convert_to_icon)
        self.convert_button.pack(padx=20, pady=20)

    def on_drop(self, event):
        self.file_path = event.data
        self.label.config(text=f"已选择文件: {self.file_path}")

    def convert_to_icon(self):
        if self.file_path:
            jpg_image = Image.open(self.file_path)
            ico_path = filedialog.asksaveasfilename(defaultextension=".ico", filetypes=[("Icon files", "*.ico")])
            if ico_path:
                jpg_image.save(ico_path)
                messagebox.showinfo("成功", f"{self.file_path} 已转为 {ico_path}")
            else:
                messagebox.showerror("错误", "请选择保存路径")
        else:
            messagebox.showerror("错误", "请先选择一个 .jpg 文件")

root = TkinterDnD.Tk()
converter = ImageToIconConverter(root)
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', converter.on_drop)
root.mainloop()
