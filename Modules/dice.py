"""

    DICE IC output management software
        Written by: Eliza Diggins


"""

### IMPORTS ###
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import logging as log
import sympy as sym
import astropy.units as u
import astropy.constants as consts
import pandas as pd
import os
import os.path
import struct
import sys
from tqdm import tqdm

try:
    import pynbody as pyn
except:
    print("WARNING: Failed to load pynbody.")

### FUNCTIONS ###

def dice_potential(filename:str,fams=None):
    ### Intro logging ###
    log.info("CMOND:Analysis:dice:dice_potential:INFO: Computing radial potential from %s."%filename)


    ### sanitizing input ###

    ## Attempting to find file ##
    if os.path.isfile(filename):
        log.debug("CMOND:Analysis:dice:dice_potential:DEBUG: Found %s. Opening."%filename)

        try:
            data = pyn.load(filename) # reading in the data.
        except Exception:
            log.error("CMOND:Analysis:dice:dice_potential:ERROR: Failed to open the data file with pynbody.")
            return False

    else:
        log.error("CMOND:Analysis:dice:dice_potential:ERROR: %s could not be found. Please check input."%filename)
        return False


    ## Family management ##
    if not fams: # no families selected for input
        fams = [fam.name for fam in data.families] # select all of them

    log.debug("CMOND:Analysis:dice:dice_potential:DEBUG: Found %s families: %s. Restricting to %s."%(len(data.families),data.families,[fam.name for fam in data.families if fam.name in fams]))

    families = [fam for fam in data.families if fam.name in fams] # restricting

    ### COMPUTING ###
    distance_array = np.sqrt(data["pos"][:,0]**2 + data["pos"][:,1]**2 + data["pos"][:,2]**2) # calculating the distance array.
def read_rz_file(filename:str):
    """
    Reads the rz file output from DICE and returns that data as a pandas dataframe.
    :param filename: The filename (path/filename) for the file in question.
    :return: Returns the full dataset as a dataframe.
    :rtype: pd.DataFrame or bool
    """
    # intro logging
    log.debug("CMOND:dice:read_rz_file:DEBUG: Reading %s."%filename)

    # Sanity check
    if not os.path.isfile(filename):
        log.error("CMOND:dice:read_rz_file:ERROR: Failed to located %s. Make sure that the filename and path are valid."%filename)
        return False

    # fetching the file
    with open(filename,"r+") as datafile:
        data = datafile.readlines()

    log.info("CMOND:dice:read_rz_file:INFO: Found %s entries in %s."%(len(data),filename))

    # managing columns
    data_columns = data[0][1:].replace("\n","").split("   ")
    data_entries = [[float(entry) for entry in row.replace("\n","").split(" ")] for row in data[1:]]

    dataframe = pd.DataFrame({column:[row[data_columns.index(column)] for row in data_entries] for column in data_columns})

    return(dataframe)

if __name__ == '__main__':
    dat = Snapshot(r"C:\Users\13852\PycharmProjects\CMOND\Datasets\DICE_IC\dice_merger.g2")
    dat.to_ascii()



