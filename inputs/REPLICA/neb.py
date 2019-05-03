# neb extract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# LAMMPS NEB format data extraction

def scatter(df, x_param, y_param, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for timestep in df.step.unique():
        if timestep == df.step.max() or timestep == df.step.min():
            l = "NEB step={}".format(timestep)
            t_dat = df[df.step == timestep].sort_values(x_param, \
                                                        ascending=True)
            ax1.plot(t_dat[x_param], t_dat[y_param], '*-', label=l)
            maxval = t_dat[y_param].max()
            for i, j in zip(t_dat[x_param], t_dat[y_param]):
                if j == maxval:
                    print(i,j)
                    ax1.annotate("{:10.4}".format(maxval),xy=(i,j), \
                            xytext=(i-.1,j-.2),arrowprops={'arrowstyle':'->'})
    plt.legend(loc='upper right')
    plt.xlabel(r'Reaction Coordinate')
    plt.ylabel(r'E [eV]')
    plt.title(title)
    plt.show()

def interactive_scatter():
    pass

def neb_extract(file):
    PE = []
    ignore_header = [0, 1, 2]
    with open(file, 'r') as l_fp:
        for i, line in enumerate(l_fp):
            if i in ignore_header or \
                    (line.startswith("Climbing") or line.startswith ("Step")):
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


def main():
    logf = sys.argv[1]
    if logf:
        try:
            data = neb_extract(logf)
        except FileNotFoundError:
            print("No file {}".format(logf))
            sys.exit(1)
    scatter(data, 'RDT', 'PE', "Cu vac migration (LAMMPS-NEB)")

if __name__ == '__main__':
    main()