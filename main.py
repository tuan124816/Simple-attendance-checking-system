import tkinter as tk
import os
from models.src.data_utils import capture_student_images, create_processed_data_directory
from models.src.train import train_new_model
from models.src.recogize import recognize


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Main Window")
        self.master.geometry('700x700')

        self.user_id = tk.StringVar()
        self.id_label = tk.Label(master, text='Register Student ID', font=('Times New Roman',10, 'bold')).place(x=50, y =150)
        self.enter_id = tk.Entry(master, textvariable=self.user_id, font=('calibre',10,'normal'),width = 30,).place(x=200, y =150)
        self.submit_bt = tk.Button(master, text='Submit', command=lambda: [self.submit_bt_push(), self.create_model()] , bd=4).place(x=420, y =150)
        self.track_bt = tk.Button(master, text='Track Face', command= self.open_subwindow, bd=4).place(x = 200, y = 235)
        self.quit_bt = tk.Button(master, text='Quit', command=master.destroy, bd = 4).place(x=200, y =270)




    def submit_bt_push(self):
        user_id = self.user_id.get()

        # NHẬP MÃ SV
        wrong_input = '\/:*?"<>|'
        while True:
            input_valid = True
            train_dir = user_id
            for char in train_dir:
                if char in wrong_input:
                    input_valid = False
                    print('Input contains invalid characters: \/:*?"<>|')
                    break
                    
            # phần trên là check input có được tạo thành folder hay không
            if input_valid == False:
                continue
            print("Input is valid.")
            # phần dưới là check folder đấy đã có chưa, có muốn thêm file/vid vào folder đấy hay không
            if os.path.exists("processed_data/" + train_dir):
                print(f"processed_data/{train_dir} already exists.")
                merge = input('input on the same folder(y/n):')
                if merge == 'y':
                    break
                else:
                    continue
            else:
                print(f"processed_data/{train_dir} is not exist.")
                # create directory
                os.mkdir("processed_data/" + train_dir)
                break

        capture_student_images(user_id)


    def create_model(self):

        if len(os.listdir('processed_data')) >= 2:
            train_new_model()
        else:
            print('need at least 2 faces')
 
    def open_subwindow(self): # This part is for depoly model here
      
        recognize()



def main():
    # create the main window
    create_processed_data_directory()
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == '__main__':
    main()
