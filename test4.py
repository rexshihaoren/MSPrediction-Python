###########################################################################
# program: tk_plot.py
# author: Tom Irvine
# Email: tom@vibrationdata.com
# version: 1.1
# date: July 6, 2013
# description:  plotting program which demonstrates Tkinter
#
###########################################################################
# 
# Note:  for use within Spyder IDE, set: 
#    
# Run > Configuration > Interpreter >
#    
# Excecute in an external system terminal
#
###########################################################################

import os
import Tkinter

import re
import numpy as np

from tkFileDialog import askopenfilename

import matplotlib.pyplot as plt

#############################################################################


def tk_read_two_columns(): 
        
    global a
    global b
        
    a = []
    b = []
    num=0
    
    input_file_path = askopenfilename(parent=root,title="Select File")

    file_path = input_file_path.rstrip('\n')
#
    if not os.path.exists(file_path):
        print("This file doesn't exist")
#
    if os.path.exists(file_path):
        print("This file exists")
        print(" ")
        infile = open(file_path,"rb")
        lines = infile.readlines()
        infile.close()


        for line in lines:
#
            if re.search(r"(\d+)", line):  # matches a digit
                iflag=0
            else:
                iflag=1 # did not find digit
#
            if re.search(r"#", line):
                iflag=1
#
            if iflag==0:
                line=line.lower()
                if re.search(r"([a-d])([f-z])", line):  # ignore header lines
                    iflag=1
                else:
                    line = line.replace(","," ")
                    col1,col2=line.split()
                    a.append(float(col1))
                    b.append(float(col2))
                    num=num+1

        a=np.array(a)
        b=np.array(b)
        
        plot_data()

        print("\n samples = %d " % num)
        

def plot_data():          
    plt.ion()
    plt.clf()
    plt.figure(1)

    index_Lbc = int(Lbc.curselection()[0]) 

    if(index_Lbc==0):
        plt.plot(a, b, linewidth=1.0,color='b')        # disregard error

    if(index_Lbc==1):
        plt.plot(a, b, linewidth=1.0,color='g')        # disregard error
        
    if(index_Lbc==2):
        plt.plot(a, b, linewidth=1.0,color='r')        # disregard error

    if(index_Lbc==3):
        plt.plot(a, b, linewidth=1.0,color='k')        # disregard error        


    plt.grid(True)
    plt.xlabel(xr.get())
    plt.ylabel(yr.get())  
    plt.title(tr.get())
    
    sx1= xmin.get()   
    sx2= xmax.get() 
 
    sy1= xmin.get()   
    sy2= xmax.get() 
   
    if sx1:
        if sx2:
            x1=float(sx1)  
            x2=float(sx2)  
            plt.xlim([x1,x2])
    
    if sy1:
        if sy2:
            y1=float(sy1)  
            y2=float(sy2)  
            plt.ylim([y1,y2])
            
    index_Lbx = int(Lbx.curselection()[0])        
    index_Lby = int(Lby.curselection()[0]) 
    
    if(index_Lbx==0 and index_Lby==0):
        plt.xscale('linear')
        plt.yscale('linear')        
    
    if(index_Lbx==0 and index_Lby==1):
        plt.xscale('linear')
        plt.yscale('log')  

    if(index_Lbx==1 and index_Lby==0):
        plt.xscale('log')
        plt.yscale('linear')  

    if(index_Lbx==1 and index_Lby==1):
        plt.xscale('log')
        plt.yscale('log')
        
        
            
    plt.draw()

    
    
def quit(root):
    root.destroy()
    

#############################################################################        

root = Tkinter.Tk()         # root (main) window   root is the parent widget
top = Tkinter.Frame(root)     # create frame
top.pack(side='top')  # pack frame in main window

root.minsize(400,300)
root.geometry("700x500")

root.title("tk_plot.py ver 1.1  by Tom Irvine")



hwtext=Tkinter.Label(top,text='The input file must have two columns')
hwtext.grid(row=0, column=0, pady=10)


button_read = Tkinter.Button(top, text="Read Input File", command=tk_read_two_columns)
button_read.config( height = 2, width = 15 )
button_read.grid(row=1, column=0, pady=20)  

################################################################################

hwtextxl=Tkinter.Label(top,text='x-axis label')
hwtextxl.grid(row=2, column=0,pady=5)

hwtextyl=Tkinter.Label(top,text='y-axis label')
hwtextyl.grid(row=2, column=1,pady=5)

hwtextti=Tkinter.Label(top,text='Title')
hwtextti.grid(row=2, column=2,padx=20,pady=5)

################################################################################

xr=Tkinter.StringVar()  
xr.set('')  
x_axis_label=Tkinter.Entry(top, width = 20,textvariable=xr)
x_axis_label.grid(row=3, column=0, pady=5)

yr=Tkinter.StringVar()  
yr.set('')   
y_axis_label=Tkinter.Entry(top, width = 20,textvariable=yr)
y_axis_label.grid(row=3, column=1,pady=5)

tr=Tkinter.StringVar()  
tr.set('')  
t_axis_label=Tkinter.Entry(top, width = 25,textvariable=tr)
t_axis_label.grid(row=3, column=2,padx=20,pady=5)

################################################################################

hwtextxmin=Tkinter.Label(top,text='x-axis lower limit')
hwtextxmin.grid(row=4, column=0,pady=5)

hwtextxmax=Tkinter.Label(top,text='x-axis upper limit')
hwtextxmax.grid(row=4, column=1,pady=5)

################################################################################

xmin=Tkinter.StringVar()  
xmin.set('')  
xmin_box=Tkinter.Entry(top, width = 20,textvariable=xmin)
xmin_box.grid(row=5, column=0, pady=5)

xmax=Tkinter.StringVar()  
xmax.set('')   
xmax_box=Tkinter.Entry(top, width = 20,textvariable=xmax)
xmax_box.grid(row=5, column=1,pady=5)

################################################################################

hwtextymin=Tkinter.Label(top,text='y-axis lower limit')
hwtextymin.grid(row=6, column=0,pady=5)

hwtextymax=Tkinter.Label(top,text='y-axis upper limit')
hwtextymax.grid(row=6, column=1,pady=5)

################################################################################

ymin=Tkinter.StringVar()  
ymin.set('')  
ymin_box=Tkinter.Entry(top, width = 20,textvariable=ymin)
ymin_box.grid(row=7, column=0, pady=5)

ymax=Tkinter.StringVar()  
ymax.set('')   
ymax_box=Tkinter.Entry(top, width = 20,textvariable=ymax)
ymax_box.grid(row=7, column=1,pady=5)

################################################################################

hwtextxtype=Tkinter.Label(top,text='x-axis type')
hwtextxtype.grid(row=8, column=0,pady=5)

hwtextytype=Tkinter.Label(top,text='y-axis type')
hwtextytype.grid(row=8, column=1,pady=5)

hwtextytype=Tkinter.Label(top,text='line color')
hwtextytype.grid(row=8, column=2,pady=5)

################################################################################


Lbx = Tkinter.Listbox(top,height=2,exportselection=0)
Lbx.insert(1, "linear")
Lbx.insert(2, "log")
Lbx.grid(row=9, column=0, pady=5)
Lbx.select_set(0) 


Lby = Tkinter.Listbox(top,height=2,exportselection=0)
Lby.insert(1, "linear")
Lby.insert(2, "log")
Lby.grid(row=9, column=1, pady=5)
Lby.select_set(0)


Lbc = Tkinter.Listbox(top,height=4,exportselection=0)
Lbc.insert(1, "blue")
Lbc.insert(2, "green")
Lbc.insert(3, "red")
Lbc.insert(4, "black")

Lbc.grid(row=9, column=2, pady=5)
Lbc.select_set(0)

################################################################################

button_plot = Tkinter.Button(top, text="Plot Data", command=plot_data)
button_plot.config( height = 2, width = 15 )
button_plot.grid(row=10, column=0, pady=20)  

button_quit=Tkinter.Button(top, text="Quit", command=lambda root=root:quit(root))
button_quit.config( height = 2, width = 15 )
button_quit.grid(row=10, column=1, padx=20,pady=20)

################################################################################

root.mainloop()   # call to event loop