import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def parseme(filename):

    tree = ET.parse(filename)

    root = tree.getroot()

    xyz = root[4][0][0][0][0].text
    l = [float(el) for el in xyz.split()]
    l=np.reshape(l,[::-1,3])
    final_l = copy.copy(l)
    if False:
        for i in range(200):
            n=222
            x = random.randint(0,n-1)
            y = random.randint(0,n-1)
            alpha = random.random()

            new_point = alpha*l[x,:] + (1-alpha)*l[y,:]
            final_l = np.vstack((final_l, new_point))
            
        for i in range(200):
            n=444
            x = random.randint(223,n-1)
            y = random.randint(223,n-1)
            alpha = random.random()

            new_point = alpha*l[x,:] + (1-alpha)*l[y,:]
            final_l = np.vstack((final_l, new_point))

        for i in range(200):
            n=666
            x = random.randint(445,n-1)
            y = random.randint(445,n-1)
            alpha = random.random()

            new_point = alpha*l[x,:] + (1-alpha)*l[y,:]
            final_l = np.vstack((final_l, new_point))

        np.save('cld',final_l)


    plt.scatter(final_l[:,1],final_l[:,2])
    plt.show()

if __name__ == "__main__":
    parseme('/home/mariem/untitled.dae')
