from tkinter import *
from tkinter import ttk
import h5py as hp
import os
import ms
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

# def calculate(*args):
#     try:
#         value = float(feet.get())
#         meters.set((0.3048 * value * 10000.0 + 0.5)/10000.0)
#     except ValueError:
#         pass

##### All the Classifiers #####
clfNames = ["LogisticRegression", "KNN", "BayesBernoulli", "BayesMultinomial", "BayesGaussian", "BayesGaussian2", "BayesPoisson", "SVM", "RandomForest", "LinearRegression", "BayesMixed", "BayesMixed2"]

##### Chooose H5 File #######
#
root = Tk()
root.title("MSPred")
root.resizable(True, True)
# root.attributes('-fullscreen', True)

# mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe = ttk.Frame(root)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
# mainframe.pack(expand =1, fill = BOTH, side = TOP)


# mainframe = ttk.Frame(root, padding="3 3 12 12", width=300,height=300).grid(column=0, row=0, sticky=(N, W, E, S))
# mainframe.columnconfigure(0, weight=1)
# mainframe.rowconfigure(0, weight=1)


####### Add Scroll Bar ######
#
#
# h = ttk.Scrollbar(root, orient=HORIZONTAL)
# v = ttk.Scrollbar(root, orient=VERTICAL)
# canvas = Canvas(root, scrollregion=(0, 0, 1000, 1000))
# # h['command'] = canvas.xview
# # v['command'] = canvas.yview
# canvas['yscrollcommand'] = v.set
# canvas['xscrollcommand'] = h.set

# ttk.Sizegrip(root).grid(column=10, row=10, sticky=(S,E))

# canvas.grid(column=0, row=0, sticky=(N,W,E,S))
# h.grid(column=10, row=1, sticky=(W,E))
# v.grid(column=1, row=10, sticky=(N,S))
#
#
# canvas = Canvas(root,width=1000,height=1000,scrollregion=(0,0,1000,1000))
# hbar=Scrollbar(root,orient=HORIZONTAL)
# hbar.pack(side=BOTTOM,fill=X)
# hbar.config(command=canvas.xview)
# vbar=ttk.Scrollbar(root,orient=VERTICAL)
# vbar.pack(side=RIGHT,fill=Y)
# vbar.config(command=canvas.yview)
# canvas.config(width=1000,height=1000)
# canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
# canvas.pack(side=LEFT,expand=True,fill=BOTH)
# 
# 
# 
# scrollbar = ttk.Scrollbar(mainframe)
# scrollbar.pack(side=RIGHT, fill=Y)

# listbox = Listbox(mainframe, yscrollcommand=scrollbar.set)
# for i in range(1000):
#     listbox.insert(END, str(i))
# listbox.pack(side=LEFT, fill=BOTH)

# scrollbar.config(command=listbox.yview)

###################

ttk.Label(mainframe, text = 'h5 file').grid(column=0, row=0, sticky=(W, E))

h5names =  [("predData", 0), ("predData_Impr0-4",2)]

h5name = StringVar()
h5name.set("predData") # initialize

for text, col in h5names:
    b = Radiobutton(mainframe, text= text,
                    variable= h5name, value=text).grid(column = col, row = 1,sticky=W)
    # b.pack(anchor=W)
objs = []
datasets = []
##### Given h5name, get datasets #####
l = Listbox(mainframe, selectmode='multiple', width = 25, height=5, exportselection = False)
s = ttk.Scrollbar(mainframe, orient=VERTICAL, command=l.yview)
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
    # for obj in objs:
    #     var = IntVar()
    #     chk = ttk.List(canvas, text=obj, variable=var).grid(column = 0, sticky = W)
    #     dsvars.append(var)


ttk.Button(mainframe, text="get datasets", command=getds).grid(column=0, row=2, sticky=W)
# h5_path = './' + h5name + '/' + h5name + '.h5'
# # print(h5_path)
# f = hp.File(h5_path, 'r')
# global objs
# objs = [str(i) for i in f.keys()]
# f.close()


# def path_finder(*args):
#     value = h5name.get()
#     h5names =  ["predData", "predData_Impr0-4"]
#     while value not in h5names:
#         h5name = raw_input("Which h5 file? do you want to use (predData or predData_Impr0-4)")
#     global general_path, h5_path, data_path, plot_path
#     general_path = './' + h5name + '/'
#     h5_path = './' + h5name + '/' + h5name + '.h5'
#     data_path = general_path + 'data/'
#     plot_path = general_path + 'plots/'
#     f = hp.File(h5_path, 'r')
#     global objs
#     objs = [str(i) for i in f.keys()]
#     f.close()

###### Display Datasets ######
#
ttk.Label(mainframe, text = 'Datasets').grid(column=0, row=3, sticky=W)


##### Display classifiers #####
clfs = []

l2 = Listbox(mainframe, selectmode='multiple', width = 25, height=5, exportselection = False)
s2 = ttk.Scrollbar(mainframe, orient=VERTICAL, command=l2.yview)
l2.grid(column=2, row=4, sticky=(N,W,E,S))
s2.grid(column=3, row=4, sticky=(N,S))
l2['yscrollcommand'] = s2.set

def getclf():
    global datasets
    datasets = [objs[i] for i in list(l.curselection())]
    clfnames = clfNames
    print("datasets: ")
    print(datasets)
    for ds in datasets:
        for clfname in clfnames:
            if not os.path.exists(data_path + ds +'/' + clfname + '_opt.h5'):
                print("remove" + clfname)
                clfnames.remove(clfname)
    print("clfnames: ")
    print(clfnames)
    for i in clfnames:
       l2.insert(END, i)


ttk.Button(mainframe, text = 'Get Classifiers', command=getclf).grid(column= 2, row = 2, sticky=W)
ttk.Label(mainframe, text = "Classifiers").grid(column= 2, row = 3, sticky=W)



####### Pic Canvas #########

def plotsd():
    # ouput figure from MSP
    global clfs
    clfs = [clfNames[i] for i in list(l2.curselection())]
    print("datasets: ")
    print(datasets)
    print("clfs: ")
    print(clfs)
    f = ms.compare_obj(datasets = datasets, models = clfs)
    a = f.add_subplot(111)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().grid(column = 0, row = 5, columnspan = 4, rowspan = 4, sticky=(N,W,E,S))
############################
# pic = Canvas(mainframe, height = 400).grid(column = 0, row = 5, columnspan = 4, rowspan = 4, sticky=(N,W,E,S))
# ttk.Button(mainframe, text = 'ROC', command=plotroc).grid(column= 4, row = 5, sticky=W)
# ttk.Button(mainframe, text = 'PR', command=plotpr).grid(column= 4, row = 6, sticky=W)
ttk.Button(mainframe, text = 'SD', command=plotsd).grid(column= 4, row = 7, sticky=W)
# ttk.Button(mainframe, text = 'Save', command=saveplot).grid(column= 4, row = 8, sticky=W)

################################



############ padding between widgets #######
for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

root.mainloop()