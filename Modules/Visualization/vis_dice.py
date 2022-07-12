"""

       Galaxy Cluster Visualizations
        Written by: Eliza Diggins


"""
### Imports
import logging as log
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Modules.dice as DICE
from tqdm import tqdm


try:
    import pynbody as pyn
except:
    print("WARNING: Failed to load pynbody on this machine. Please use linux.")


## Sub-Functions ##
def get_random_color(n):
    """
    Grabs n random colors from the hsv colormap

    :param n: The number of colors to grab.
    :return: Returns the colors.
    """
    map = plt.cm.get_cmap("hsv", n)  # grabbing the map

    return [map(i) for i in range(n)]


### Functions ###

def plot_particles(filename, axes=None, echo=True, color_dict=None, sampling=10, alpha=0.2, fams=None,size=0.01):
    """
    Plots the locations of the particles in the given file.
    :param filename: The filename to open.
    :param axes: The axes to use. If None, then generates new axes.
    :param echo: True to echo output, False to silence.
    :param color_dict: The dictionary for colors. should be {family_name:color}. Auto generates.
    :param sampling: How frequently to sample each set. Higher means less particles.
    :param alpha: The opacity of the points.
    :param fams: The families to include.
    :return: axes.
    """

    ### open the file ###
    try:
        data = pyn.load(filename)

        if echo:
            log.debug("CMOND:Visualization:vis_dice:plot_particles:DEBUG: Loaded %s: %s." % (filename, data))
    except OSError:
        raise SyntaxError("The filename %s was not recognized, or was not found." % filename)

    # grabbing families
    families = data.families()

    # debug
    if echo:
        log.debug("CMOND:Visualization:vis_dice:plot_particles:DEBUG: Found %s familes: %s" % (len(families), families))

    # managing families
    if fams:
        families = [fam for fam in families if fam.name in fams]
        if echo:
            log.debug("CMOND:Visualization:vis_dice:plot_particles:DEBUG: Restricted families to %s." % families)

    # managing the color dictionary
    if not color_dict:
        if echo:
            log.debug("CMOND:Visualization:vis_dice:plot_particles:DEBUG: No color_dict found. Constructing.")

        tmp_colors = get_random_color(len(families))
        color_dict = {families[i]: tmp_colors[i] for i in range(len(families))}

    else:
        if echo:
            log.debug("CMOND:Visualization:vis_dice:plot_particles:DEBUG: Found color_dict = %s" % color_dict)

    ### PLOTTING ###
    if not axes:
        if echo:
            log.debug("CMOND:Visualization:vis_dice:plot_particles:DEBUG: No axes specified. Creating axes.")

        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
    else:
        if echo:
            log.debug("CMOND:Visualization:vis_dice:plot_particles:DEBUG: Axes inputted manually: %s." % axes)

    for family in families:  # plotting each family
        if echo:
            log.debug("CMOND:Visualization:vis_dice:plot_particles:DEBUG: Plotting %s particles. n=%s." % (family.name, len(data[family])))

        pos_data = data[family]['pos']

        xs = [i[0] for i in pos_data[::sampling]]
        ys = [i[1] for i in pos_data[::sampling]]
        zs = [i[2] for i in pos_data[::sampling]]

        axes.scatter(xs, ys, zs, color=color_dict[family], alpha=alpha, label=family.name,size=size)

    ### managing axes labels ###
    axes.set_ylabel("y (%s)" % data["pos"].units)
    axes.set_zlabel("z (%s)" % data["pos"].units)
    axes.set_xlabel("x (%s)" % data["pos"].units)
    axes.set_title("Particles In %s" % filename)

    return axes


def plot_density(filename,
                 axes=None,
                 orientation="xy",
                 cross_section=0,
                 cross_height=2,
                 echo=True,
                 cmap=plt.cm.Greens,
                 n=10,
                 fams=None,
                 pad=0.1,
                 contours=True,
                 cont_cmap=plt.cm.inferno):
    """
    Plots the density of the cross section.
    :param filename:
    :param axes:
    :param cross_section:
    :param echo:
    :param cmap:
    :param sampling:
    :param fams:
    :return:
    """
    ### Reading data ###
    try:  # pull the data if it exists.
        data = pyn.load(filename)
    except OSError:  # Failed to find the data
        log.error("CMOND:Visualization:vis_dice:plot_density:ERROR: Failed to load the file %s." % filename)
        return False

    ### Sanitizing ###

    # orientation
    if orientation not in ["xy", "yx", "zx", "xz", "zy", "yz"]:  # the orientation is invalid
        log.error("CMOND:Visualization:vis_dice:plot_density:ERROR: %s is not a valid orientation. Please select from %s." % (
            orientation, ["xy", "yx", "zx", "xz", "zy", "yz"]))
    elif orientation in ["yx", "zx", "zy"]:
        orientation = orientation[1] + orientation[0]  # Simply reversing the orientation.

    # cross section
    if not isinstance(cross_section, (float, int)):  # the dtype is wrong.
        log.error("CMOND:Visualization:vis_dice:plot_density:ERROR: %s is of type %s, not type %s; therefore, it is not a valid choice of cross_section." % (
            cross_section, type(cross_section), float))
    if not isinstance(cross_height, (float, int)):  # the dtype is wrong.
        log.error("CMOND:Visualization:vis_dice:plot_density:ERROR: %s is of type %s, not type %s; therefore, it is not a valid choice of cross_height." % (
            cross_height, type(cross_height), float))

    ### Managing Fams ###
    families = data.families()  # reading the families.

    if fams:
        families = [family for family in families if family.name in fams]
    else:
        fams = [fam.name for fam in families]

    if echo:
        log.debug("CMOND:Visualization:vis_dice:plot_density:DEBUG: Familes found for %s: %s. %s are used." % (
            filename, [i.name for i in families], [i.name for i in families if i.name in fams]))

    ### Constructing grid ###

    ## grabbing min,max ##
    if orientation == 'xy':  # we are using xy orientation
        ors = [0,1,2]
        bounds = [(1 + pad) * np.amin(data["pos"][:, 0]),
                  (1 + pad) * np.amax(data["pos"][:, 0]),
                  (1 + pad) * np.amin(data["pos"][:, 1]),
                  (1 + pad) * np.amax(data["pos"][:, 1])
                  ]
    elif orientation == 'xz':
        ors = [0,2,1]
        bounds = [(1 + pad) * np.amin(data["pos"][:, 0]),
                  (1 + pad) * np.amax(data["pos"][:, 0]),
                  (1 + pad) * np.amin(data["pos"][:, 2]),
                  (1 + pad) * np.amax(data["pos"][:, 2])
                  ]
    else:
        ors = [1,2,0]
        bounds = [(1 + pad) * np.amin(data["pos"][:, 1]),
                  (1 + pad) * np.amax(data["pos"][:, 1]),
                  (1 + pad) * np.amin(data["pos"][:, 2]),
                  (1 + pad) * np.amax(data["pos"][:, 2])
                  ]

    if echo:
        log.debug("CMOND:Visualization:vis_dice:plot_density:DEBUG: Bounds are computed to be %s:[%s,%s], %s:[%s,%s]" % (
            orientation[0], bounds[0], bounds[1], orientation[1], bounds[2], bounds[3]))

    ## Constructing bounds ##
    d1s = np.linspace(bounds[0], bounds[1], n)  # building the array.
    d2s = np.linspace(bounds[2], bounds[3], n)

    ## building the array ##
    matrix = np.zeros((n, n))

    if echo:
        log.debug("CMOND:Visualization:vis_dice:plot_density:DEBUG: Constructed empty matrix of size %sx%s."%(n,n))

    or1 = ors[0]
    or2 = ors[1]
    or3 = ors[2]

    pos_set = data["pos"][(cross_section-cross_height<=data["pos"][:,or3])&(data["pos"][:,or3]<=cross_section+cross_height)]
    pos_mass = data["mass"][(cross_section-cross_height<=data["pos"][:,or3])&(data["pos"][:,or3]<=cross_section+cross_height)]


    for id,pos in enumerate(tqdm(data["pos"],desc="Computing Matrix")):
        matrix[len(d1s[d1s<pos[or1]])-1,len(d2s[d2s<pos[or2]])-1] += data["mass"][id]

    if not axes:
        fig = plt.figure()
        axes = fig.add_subplot(111)

    im = axes.imshow(np.log(matrix),cmap=cmap,origin="lower",extent=bounds)

    if contours:
        if echo:
            log.debug("CMOND:Visualization:vis_dice:plot_density:DEBUG: Contours being added with colormap: %s."%cont_cmap)
        # We are going to add contour lines
        axes.contour(d1s,d2s,np.log(matrix),cmap=cont_cmap) # add the contours
    else:
        pass

    ### Cleanup ###
    if echo:
        log.debug("CMOND:Visualization:vis_dice:plot_density:DEBUG: Cleaning up...")

    axes.set_title(r"Mass Distribution Profile From %s"%filename.split("/")[-1])
    axes.set_xlabel(r"%s [%s]"%(orientation[0],str(data["pos"].units)[-3:]))
    axes.set_ylabel(r"%s [%s]" % (orientation[1], str(data["pos"].units)[-3:]))

    cbar = plt.colorbar(im,ax=axes)
    cbar.ax.set_ylabel(r"$\log\left(M(%s,%s)\right)$ [%s]"%(orientation[0],orientation[1],data["mass"].units))

def plot_rz_densities(filenames, bounds: list = None, resolution: int = 1,save=True,end_file=None,cmap=plt.cm.Greens) -> bool:
    """
    Plots the density provided by the rz files within bounds.
    :param filenames: The filenames and path to the given file.
    :type filenames: list[str]
    :param bounds: The bounds on r and z in kpc [r_min,r_max,z_min,z_max]. Defaults to the entire dataset.
    :param resolution: The resolution (number of points per box) to plot at. Min is 1, max is the length of the dataset.
    :return: None
    """
    # intro logging
    log.debug("CMOND:Visualization:vis_dice:vis_dice:plot_rz_densities:DEBUG: Plotting density from %s with bounds %s." % (filenames, bounds))

    ### SANITY CHECKING ###

    # Checking for singularity
    if isinstance(filenames, (str)):  # The filenames input was singular, so we need to make a list.
        filenames = [filenames]  # creating the list form.

    # Checking for false types
    if not isinstance(filenames, list):  # the filenames are not in a list at this point, something is wrong.
        log.error(
            "CMOND:Visualization:vis_dice:vis_dice:plot_rz_density:ERROR: %s is not a valid type for input 'filenames'. Please try again." % type(
                filenames))
        return False

    # Checking for invalid filenames
    if not all(os.path.isfile(filename) for filename in filenames):  # At least one of the filenames is invalid.
        failed_files = [file for file in filenames if not os.path.isfile(file)]  # Create a list of the invalid files.
        filenames = [file for file in filenames if file not in failed_files]

        # logging
        log.error(
            "CMOND:Visualization:vis_dice:vis_dice:plot_rz_density:ERROR: Failed to find %s. Please check the location and spelling. Removing." % failed_files)
    else:
        # Nothing is wring
        pass

    ### BUILDING THE DATASET ###

    dataframe = pd.DataFrame({})  # create a blank dataframe.

    FAILURE_COUNT = 0  # counter for failures.
    for file in filenames:  # search through all of the files
        log.debug("CMOND:Visualization:vis_dice:vis_dice:plot_rz_densities:DEBUG: Attempting to fetch data for %s." % file)

        # reading the data
        temp_frame = DICE.read_rz_file(file)

        if type(temp_frame) is not bool:  # The file was analyzed sucessfully
            dataframe = pd.concat([dataframe, temp_frame], ignore_index=True)
        else:
            log.warning("CMOND:Visualization:vis_dice:vis_dice:plot_rz_densities:WARNING: Failed to load data from %s. Passing." % file)

    log.info(
        "CMOND:Visualization:vis_dice:vis_dice:plot_rz_densities:INFO: Finished data read. Sucesses: %s, Failures %s (%% %s). Dataframe:\n%s" % (
        len(filenames) - FAILURE_COUNT, FAILURE_COUNT, 100 * (FAILURE_COUNT / len(filenames)), dataframe.to_string()))

    ### MANIPULATION ###

    # Axis management

    rs = list(set(list(dataframe["r"])))  # sorting and removing duplicates
    zs = list(set(list(dataframe["z"])))

    ## adding negatives

    rs += [-1 * i for i in rs]
    zs += [-1 * i for i in zs]

    ## sorting

    rs = sorted(rs)
    zs = sorted(zs)

    if not bounds:  # the bounds haven't been specified.
        bounds = [np.amin(rs), np.amax(rs), np.amin(zs), np.amax(rs)]

    # Removing excess
    rs = [element for element in rs if bounds[0] <= element <= bounds[1]]
    zs = [element for element in zs if bounds[2] <= element <= bounds[3]]

    ### building array
    density_profile = np.array([[sum(dataframe.loc[(dataframe["r"].between(rs[i], rs[i + resolution - 1],
                                                                           inclusive="left")|(-1*dataframe["r"]).between(rs[i], rs[i + resolution - 1],
                                                                           inclusive="left")) & (dataframe["z"].between(zs[j], zs[j + resolution - 1],
                                                                           inclusive="left")|(-1*dataframe["z"]).between(zs[j], zs[j + resolution - 1],
                                                                           inclusive="left")), "rho"].values) for i
                                 in np.arange(0, len(rs) - resolution, resolution)] for j in
                                np.arange(0, len(zs) - resolution, resolution)])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    h = ax.imshow(density_profile, extent=bounds,cmap=cmap)

    ### Adding information
    ax.set_title("Mass Density of Initial Conditions")
    ax.set_xlabel(r"Distance from System Center $x$ [kpc]")
    ax.set_ylabel(r"Distance from System Center $y$ [kpc]")

    plt.colorbar(h,ax=ax)

    if save:
        if end_file:
            os.chdir(end_file)
            plt.savefig("init_cons.png")
        else:
            plt.savefig("init_cons.png")
    else:
        plt.show()



if __name__ == '__main__':
    log.basicConfig(level=log.DEBUG)
    plot_density("")
