from tkinter import *
from tkinter import ttk
import h5py as hp
import os
import ms
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import pylab as pl
from tkinter import tix
class AutoScrollbar(Scrollbar):
    # a scrollbar that hides itself if it's not needed.  only
    # works if you use the grid geometry manager.
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("grid", "remove", self)
        else:
            self.grid()
        Scrollbar.set(self, lo, hi)
    def pack(self, **kw):
        raise TclError("cannot use pack with this widget")
    def place(self, **kw):
        raise TclError("cannot use place with this widget")


root = Tk()
root.title("MSPred")
root.geometry("1000x1000")
vscrollbar = AutoScrollbar(root)
vscrollbar.grid(row=0, column=1, sticky=N+S)
hscrollbar = AutoScrollbar(root, orient=HORIZONTAL)
hscrollbar.grid(row=1, column=0, sticky=E+W)

canvas = Canvas(root, scrollregion=(0, 0, 1000, 1000),
                yscrollcommand=vscrollbar.set,
                xscrollcommand=hscrollbar.set)
canvas.grid(row=0, column=0, sticky=N+S+E+W)

vscrollbar.config(command=canvas.yview)
hscrollbar.config(command=canvas.xview)

# make the canvas expandable
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

#
# create canvas contents

frame = Frame(canvas)
frame.rowconfigure(1, weight=1)
frame.columnconfigure(1, weight=1)



########## MY OWN #####################






##### All the Classifiers #####
clfNames = ["LogisticRegression", "KNN", "BayesBernoulli", "BayesMultinomial", "BayesGaussian", "BayesGaussian2", "BayesPoisson", "SVM", "RandomForest", "LinearRegression", "BayesMixed", "BayesMixed2"]


ttk.Label(frame, text = 'h5 file').grid(column=0, row=0, sticky=(W, E))

h5names =  [("predData", 0), ("predData_Impr0-4",2)]

h5name = StringVar()
h5name.set("predData") # initialize

for text, col in h5names:
    b = Radiobutton(frame, text= text,
                    variable= h5name, value=text).grid(column = col, row = 1,sticky=W)
    # b.pack(anchor=W)
objs = []
datasets = []
##### Given h5name, get datasets #####
l = Listbox(frame, selectmode='multiple', width = 25, height=5, exportselection = False)
s = ttk.Scrollbar(frame, orient=VERTICAL, command=l.yview)
l.grid(column=0, row=4, sticky=(N,W,E,S))
s.grid(column=1, row=4, sticky=(N,S))
l['yscrollcommand'] = s.set
dsvars = []
plot_path = " "

def getds():
    value = h5name.get()
    global general_path, h5_path, data_path, plot_path
    general_path = './' + value + '/'
    h5_path = './' + value + '/' + value + '.h5'
    data_path = general_path + 'data/'
    plot_path = general_path + 'plots/'
    ms.plot_path = plot_path
    ms.data_path = data_path 
    f = hp.File(h5_path, 'r')
    global objs
    objs = [str(i) for i in f.keys()]
    f.close()
    for i in objs:
       l.insert(END, i)


ttk.Button(frame, text="Get Datasets", command=getds).grid(column=0, row=2, sticky=W)


###### Display Datasets ######
#
ttk.Label(frame, text = 'Datasets').grid(column=0, row=3, sticky=W)


##### Display classifiers #####
clfs = []

l2 = Listbox(frame, selectmode='multiple', width = 25, height=5, exportselection = False)
s2 = ttk.Scrollbar(frame, orient=VERTICAL, command=l2.yview)
l2.grid(column=2, row=4, sticky=(N,W,E,S))
s2.grid(column=3, row=4, sticky=(N,S))
l2['yscrollcommand'] = s2.set

def getclf():
    global datasets
    datasets = [objs[i] for i in list(l.curselection())]
    print("datasets: ")
    print(datasets)
    global newclfNames
    newclfNames = clfNames.copy()
    for ds in datasets:
        for clfname in clfNames:
            if (not os.path.exists(data_path + ds +'/' + clfname + '_opt.h5')) and (clfname in newclfNames):
                print("remove" + clfname)
                newclfNames.remove(clfname)
    for i in newclfNames:
       l2.insert(END, i)


ttk.Button(frame, text = 'Get Classifiers', command=getclf).grid(column= 2, row = 2, sticky=W)
ttk.Label(frame, text = "Classifiers").grid(column= 2, row = 3, sticky=W)


####### Pic Canvas #########
f = pl.figure(figsize=(4,4),dpi=100)
pic = FigureCanvasTkAgg(f, master=frame)
pic.show()
pic.get_tk_widget().grid(column = 0, row = 5, columnspan = 4, rowspan = 4, sticky=(N,W,E,S))
### Save Path #####
p = " "
def plotopt(tp):
    # ouput figure from ms
    global clfs
    clfs = [newclfNames[i] for i in list(l2.curselection())]
    print("datasets: ")
    print(datasets)
    print("clfs: ")
    print(clfs)
    global f, p
    f= ms.compare_wrappers(datasets = datasets, models = clfs, opt = True, tp = tp)
    p = plot_path + tp
    for i in datasets:
        p = p + '_' + i
    for j in clfs:
        p = p + '_' + j
    p = p + '.pdf'
    global pic
    pic = FigureCanvasTkAgg(f, master=frame)
    pic.show()
    pic.get_tk_widget().grid(column = 0, row = 5, columnspan = 4, rowspan = 4, sticky=(N,W,E,S))
    # canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


def saveplot():
    f.savefig(p)

############################
# pic = Canvas(frame, height = 400).grid(column = 0, row = 5, columnspan = 4, rowspan = 4, sticky=(N,W,E,S))
ttk.Button(frame, text = 'ROC', command=lambda: plotopt('roc_auc')).grid(column= 4, row = 5, sticky=W)
ttk.Button(frame, text = 'PR', command=lambda:plotopt('pr')).grid(column= 4, row = 6, sticky=W)
ttk.Button(frame, text = 'SD', command=lambda:plotopt('sd')).grid(column= 4, row = 7, sticky=W)
ttk.Button(frame, text = 'Save', command=saveplot).grid(column= 4, row = 8, sticky=W)

################################

def reset():
    l.delete(0,END)
    l2.delete(0,END)
    f = pl.figure(figsize=(4,4),dpi=100)
    pic = FigureCanvasTkAgg(f, master=frame)
    pic.show()
    pic.get_tk_widget().grid(column = 0, row = 5, columnspan = 4, rowspan = 4, sticky=(N,W,E,S))

ttk.Button(frame, text = 'Reset', command=reset).grid(column= 2, row = 0, sticky=W)

############ padding between widgets #######
for child in frame.winfo_children(): child.grid_configure(padx=5, pady=5)





#############################################





canvas.create_window(0, 0, anchor=NW, window=frame)

frame.update_idletasks()

canvas.config(scrollregion=canvas.bbox("all"))

root.mainloop()