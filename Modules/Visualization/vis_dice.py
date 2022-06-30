"""

       Galaxy Cluster Visualizations
        Written by: Eliza Diggins


"""

### Imports
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import logging as log
import sympy as sym
import Modules.dice as DICE
import os
from datetime import datetime
import gc
import matplotlib as mpl

def plot_rz_density(filename,bounds=None,resolution=1):
    """
    Plots the density provided by the rz file within bounds.
    :param filename: The filename and path to the given file.
    :param bounds: The bounds on r and z in kpc [r_min,r_max,z_min,z_max]. Defaults to the entire dataset.
    :param resolution: The resolution (number of points per box) to plot at. Min is 1, max is the length of the dataset.
    :return: None
    """
    # intro logging
    log.debug("CMOND:vis_dice:plot_rz_density:DEBUG: Plotting density from %s with bounds %s."%(filename,bounds))

    # sanity checking and kwarg management
    if not os.path.isfile(filename): # the filename is invalid
        log.error("CMOND:vis_dice:plot_rz_density:ERROR: Failed to find %s. Please check the location and spelling."%filename)

    ### grabbing the dataset
    dataframe = DICE.read_rz_file(filename)

    ### cleanup
    rs = list(set(list(dataframe["r"]))) # sorting and removing duplicates
    zs = list(set(list(dataframe["z"])))
    ## adding negatives

    rs += [-1*i for i in rs]
    zs += [-1*i for i in zs]

    ## sorting
    rs = sorted(rs)
    zs = sorted(zs)



    if not bounds: # the bounds haven't been specified.
        bounds = [np.amin(rs),np.amax(rs),np.amin(zs),np.amax(rs)]

    # Removing excess
    rs = [element for element in rs if bounds[0]<=element<=bounds[1]]
    zs = [element for element in zs if bounds[2]<=element<=bounds[3]]

    ### building array
    density_profile = np.array([[dataframe.loc[(dataframe["r"]==np.abs(r))&(dataframe["z"]==np.abs(z)),"rho"].item() for r in rs[::resolution]] for z in zs[::resolution]])

    plt.imshow(density_profile)
    plt.show()

if __name__ == '__main__':
    plot_rz_density(r"C:\Users\13852\PycharmProjects\CMOND\Datasets\DICE_IC\galaxy1.params.rz5",resolution=2)


