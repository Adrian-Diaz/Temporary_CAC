# neb extract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LAMMPS NEB format data extraction

def scatter(df, x_param, y_param, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for timestep in df.step.unique():
        if timestep == df.step.max() or timestep == df.step.min():
            l = "NEB step={}".format(timestep)
            t_dat = df[df.step == timestep].sort_values(x_param, ascending=True)
            ax1.plot(t_dat[x_param], t_dat[y_param], '*-', label=l)
            maxval = t_dat[y_param].max()
            for i, j in zip(t_dat[x_param], t_dat[y_param]):
                if j == maxval:
                    print(i,j)
                    ax1.annotate("{:10.4}".format(maxval),xy=(i,j), xytext=(i-.1,j-.2),arrowprops={'arrowstyle':'->'})
    plt.legend(loc='upper right')
    plt.xlabel(r'Reaction Coordinate')
    plt.ylabel(r'E [eV]')
    plt.title(title)
    plt.show()

def interactive_scatter():
    pass

def neb_extract(file):
    PE = []
    ignore = [0, 1, 2, 6, 7]
    with open(file, 'r') as l_fp:
        for i, line in enumerate(l_fp):
            if i in ignore:
                pass
            else:
                l = (line.split())
                info = l[0:9]
                t = info[0]
                j = 9
                firstE = float(l[j+1])
                while j < len(l)-1:
                #    print(float(l[j+1])-firstE)
                    PE.append([int(t), float(l[j]), float(l[j+1])-firstE])
                    j += 2

    df = pd.DataFrame(PE, columns=['step', 'RDT', 'PE'])
    return df



def b():
    data = neb_extract('log.neb')
    scatter(data, 'RDT', 'PE', "Ni Dislocation-Vacancy Interaction")

b()