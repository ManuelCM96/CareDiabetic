from email.mime import application
from tkinter import ttk
from tkinter import *
import sqlite3
class Prueba:
    def __init__(self,window):
        self.wind=window
        self.wind.title('CareDiabetic')
        ##Crear un Contenedor
        frame=LabelFrame(self.wind,text='Demo de Obtencion de Datos del Paciente')
        frame.grid(row=0,column=0,columnspan=3,pady=20)
        #
        Label(frame,text='Name:  ').grid(row=1,column=0)
if __name__=='__main__':
    window=Tk()
    application=Prueba(window)
    window.mainloop()