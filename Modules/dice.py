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


### FUNCTIONS ###
def read_rz_file(filename:str)->pd.DataFrame:
    """
    Reads the rz file output from DICE and returns that data as a pandas dataframe.
    :param filename: The filename (path/filename) for the file in question.
    :return: Returns the full dataset as a dataframe.
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
    dat = read_rz_file(r"C:\Users\13852\PycharmProjects\CMOND\Datasets\DICE_IC\galaxy1.params.rz1")



