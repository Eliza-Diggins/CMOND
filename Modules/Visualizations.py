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
import Modules.Clusters as C

### Functions

def plot_temperature_profiles(clusters,
                              r_min=None,
                              r_max=None,
                              n=500,
                              indp_unit = u.kpc,
                              dep_unit = u.keV,
                              scale="semilogx"
                              ):
    """
    Plots the temperature profiles for the given clusters.
    :param clusters: The list of cluster objects to plot.
    :type clusters: list[Cluster]
    :return:
    """
    log.info("CMOND:Visualizations:plot_temperature_profiles:INFO plotting temperature profiles for %s."%[cluster.name for cluster in clusters])

    # Cleaning clusters
    USABLE_CLUSTERS = [cluster for cluster in clusters if cluster._has_temp_profile]
    log.debug("CMOND:Visualization:plot_temperature_profiles:DEBUG: Usable clusters are %s"%[cluster.name for cluster in USABLE_CLUSTERS])
    
    # Fixing bounds
    if r_min == None:
        r_min = min([c.r_min for c in USABLE_CLUSTERS if c.r_min != None])
        CUSTOM_MIN=False
    else:
        CUSTOM_MIN = True
    if r_max == None:
        r_max = max([c.r_500 for c in USABLE_CLUSTERS if c.r_500 != None])
        CUSTOM_MAX = False
    else:
        CUSTOM_MAX = True

    r_min = r_min.to(indp_unit)
    r_max = r_max.to(indp_unit)
    
    
    # plotting
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(111)

    # Managing the scale

    if scale == "semilogx":
        ax1.set_xscale("log")
    elif scale == "semilogy":
        ax1.set_yscale("log")
    elif scale == "loglog":
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    else:
        pass


    
    for cluster in USABLE_CLUSTERS:
        # Building r array
        if not CUSTOM_MAX:
            if cluster.r_500 != None:
                ar_max = cluster.r_500
            else:
                ar_max = r_max
        else:
            ar_max = r_max
        
        if not CUSTOM_MIN:
            if cluster.r_min != None:
                ar_min = cluster.r_min
            else:
                ar_min = r_min
        else:
            ar_min = r_min

        r_array = np.linspace(ar_min,ar_max,n)

        output = sym.lambdify(sym.Symbol("r"),cluster.temp_profile,"numpy")((r_array).to(C._CMOND_base_units["m"]).value)
        corrected_output = (output*cluster._temp_profile_output_units).to(dep_unit).value


        ax1.plot(r_array,corrected_output  ,label=cluster.name)

    ax1.set_xlabel("Radial Distance [%s]"%str(indp_unit))
    ax1.set_ylabel("Gas Temperature [%s]"%str(dep_unit))
    ax1.set_title("Gas Temperature Profiles")
    plt.subplots_adjust(left=0.1,right=0.75)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05,1),title="Clusters")

    plt.show()


def plot_density_profiles(clusters,
                              r_min=None,
                              r_max=None,
                              n=500,
                              indp_unit=u.kpc,
                              dep_unit=u.solMass/(u.m**3),
                              outer_scale_factor=1.0,
                              scale="semilogx"
                              ):
    """
    Plots the temperature profiles for the given clusters.
    :param clusters: The list of cluster objects to plot.
    :type clusters: list[Cluster]
    :return:
    """
    log.info(
        "CMOND:Visualizations:plot_temperature_profiles:INFO plotting temperature profiles for %s." % [cluster.name for
                                                                                                       cluster in
                                                                                                       clusters])

    # Cleaning clusters
    USABLE_CLUSTERS = [cluster for cluster in clusters if cluster._has_density_profile]
    log.debug(
        "CMOND:Visualization:plot_temperature_profiles:DEBUG: Usable clusters are %s" % [cluster.name for cluster in
                                                                                         USABLE_CLUSTERS])

    # Fixing bounds
    if r_min == None:
        r_min = min([c.r_min for c in USABLE_CLUSTERS if c.r_min != None])
        CUSTOM_MIN = False
    else:
        CUSTOM_MIN = True
    if r_max == None:
        r_max = max([c.r_500 for c in USABLE_CLUSTERS if c.r_500 != None])
        CUSTOM_MAX = False
    else:
        CUSTOM_MAX = True

    r_min = r_min.to(indp_unit)
    r_max = r_max.to(indp_unit)

    # plotting
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    # Managing the scale

    if scale == "semilogx":
        ax1.set_xscale("log")
    elif scale == "semilogy":
        ax1.set_yscale("log")
    elif scale == "loglog":
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    else:
        pass

    for cluster in USABLE_CLUSTERS:
        # Building r array
        if not CUSTOM_MAX:
            if cluster.r_500 != None:
                ar_max = cluster.r_500
            else:
                ar_max = r_max
        else:
            ar_max = r_max

        if not CUSTOM_MIN:
            if cluster.r_min != None:
                ar_min = cluster.r_min
            else:
                ar_min = r_min
        else:
            ar_min = r_min

        r_array = np.linspace(ar_min,ar_max,n)

        output = sym.lambdify(sym.Symbol("r"), cluster.density_profile, "numpy")(
            r_array.to(C._CMOND_base_units["m"]).value)
        corrected_output = (output * cluster._density_profile_output_units).to(dep_unit).value


        ax1.plot(r_array, corrected_output, label=cluster.name)

    ax1.set_xlabel("Radial Distance [%s]" % str(indp_unit))
    ax1.set_ylabel("Gas Density [%s]" % str(dep_unit))
    ax1.set_title("Gas Density Profiles")
    plt.subplots_adjust(left=0.1, right=0.75)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.1, 1), title="Clusters")

    plt.show()

def plot_dynamical_mass_profiles(clusters,
                              r_min=None,
                              r_max=None,
                              n=500,
                              indp_unit=u.kpc,
                              dep_unit=u.solMass,
                              scale="semilogx",
                              mode="NEWT"
                              ):
    """
    Plots the temperature profiles for the given clusters.
    :param clusters: The list of cluster objects to plot.
    :type clusters: list[Cluster]
    :return:
    """
    log.info(
        "CMOND:Visualizations:plot_temperature_profiles:INFO plotting temperature profiles for %s." % [cluster.name for
                                                                                                       cluster in
                                                                                                       clusters])

    # Cleaning clusters
    USABLE_CLUSTERS = [cluster for cluster in clusters if cluster._has_dynamical_mass_profile_newt]
    log.debug(
        "CMOND:Visualization:plot_temperature_profiles:DEBUG: Usable clusters are %s" % [cluster.name for cluster in
                                                                                         USABLE_CLUSTERS])

    # Fixing bounds
    if r_min == None:
        r_min = min([c.r_min for c in USABLE_CLUSTERS if c.r_min != None])
        CUSTOM_MIN = False
    else:
        CUSTOM_MIN = True
    if r_max == None:
        r_max = max([c.r_500 for c in USABLE_CLUSTERS if c.r_500 != None])
        CUSTOM_MAX = False
    else:
        CUSTOM_MAX = True

    r_min = r_min.to(indp_unit)
    r_max = r_max.to(indp_unit)

    # plotting
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    # Managing the scale

    if scale == "semilogx":
        ax1.set_xscale("log")
    elif scale == "semilogy":
        ax1.set_yscale("log")
    elif scale == "loglog":
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    else:
        pass

    for cluster in USABLE_CLUSTERS:
        # Building r array
        if not CUSTOM_MAX:
            if cluster.r_500 != None:
                ar_max = cluster.r_500
            else:
                ar_max = r_max
        else:
            ar_max = r_max

        if not CUSTOM_MIN:
            if cluster.r_min != None:
                ar_min = cluster.r_min
            else:
                ar_min = r_min
        else:
            ar_min = r_min

        r_array = np.linspace(ar_min,ar_max,n)

        if mode == "NEWT":
            output = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_newt, "numpy")(
            (r_array).to(C._CMOND_base_units["m"]).value)
            corrected_output = (output * cluster._dynamical_mass_profile_newt_units).to(dep_unit).value
        elif mode == "MOND":
            output = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_mond, "numpy")(
            (r_array).to(C._CMOND_base_units["m"]).value)
            corrected_output = (output * cluster._dynamical_mass_profile_mond_units).to(dep_unit).value



        ax1.plot(r_array, corrected_output, label=cluster.name)

    ax1.set_xlabel("Radial Distance [%s]" % str(indp_unit))
    ax1.set_ylabel(r"Dynamical Mass In Radius $r$ [%s]" % str(dep_unit))
    if mode == "NEWT":
        ax1.set_title("Newtonian Dynamical Mass Profiles")
    elif mode == "MOND":
        ax1.set_title("Milgromian Dynamical Mass Profiles")
    plt.subplots_adjust(left=0.1, right=0.75)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.1, 1), title="Clusters")

    plt.show()

def plot_mond_newt_profile(cluster,r_min=None,r_max=None,indp_unit=u.kpc,scale="loglog",dep_unit=u.solMass,n=1000):
    """
    Plots both the milgromian and newtonian profiles for the mass.
    :param cluster: The cluster to plot
    :return: None
    """
    log.info(
        "CMOND:Visualizations:plot_mond_newt_profile:INFO plotting mond and newt profiles for %s." % cluster.name)

    if not (cluster._has_dynamical_mass_profile_newt and cluster._has_dynamical_mass_profile_mond):
        raise NotImplementedError("CMOND:Visualizations:plot_mond_newt_profile:ERROR: Failed to find necessary profiles for plotting.")

    # Fixing bounds
    if r_min == None:
        try:
            r_min = cluster.r_min
        except Exception:
            r_min = 0*u.kpc
        CUSTOM_MIN = False
    else:
        CUSTOM_MIN = True
    if r_max == None:
        try:
            r_max = cluster.r_500
        except Exception:
            r_max = 1000*u.kpc
        CUSTOM_MAX = False
    else:
        CUSTOM_MAX = True

    r_min = r_min.to(indp_unit)
    r_max = r_max.to(indp_unit)

    # plotting
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    # Managing the scale

    if scale == "semilogx":
        ax1.set_xscale("log")
    elif scale == "semilogy":
        ax1.set_yscale("log")
    elif scale == "loglog":
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    else:
        pass

    if not CUSTOM_MAX:
        if cluster.r_500 != None:
            ar_max = cluster.r_500
        else:
            ar_max = r_max
    else:
        ar_max = r_max
    if not CUSTOM_MIN:
        if cluster.r_min != None:
            ar_min = cluster.r_min
        else:
            ar_min = r_min
    else:
        ar_min = r_min
    r_array = np.linspace(ar_min, ar_max, n)

    output_newt = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_newt, "numpy")(
            (r_array).to(C._CMOND_base_units["m"]).value)
    output_mond = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_mond, "numpy")(
            (r_array).to(C._CMOND_base_units["m"]).value)



    corrected_output_newt = (output_newt * cluster._dynamical_mass_profile_newt_units).to(dep_unit).value
    corrected_output_mond = (output_mond * cluster._dynamical_mass_profile_mond_units).to(dep_unit).value



    ax1.plot(r_array, corrected_output_mond, label="MOND")
    ax1.plot(r_array,corrected_output_newt,label="NEWT")

    ax1.set_xlabel("Radial Distance [%s]" % str(indp_unit))
    ax1.set_ylabel(r"Dynamical Mass In Radius $r$ [%s]" % str(dep_unit))
    ax1.set_title("Dynamical Masses of %s in MOND and Newtonian Paradigms"%cluster.name)
    plt.subplots_adjust(left=0.1, right=0.75)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.1, 1), title="Clusters")

    plt.show()

if __name__ == '__main__':
    log.basicConfig(level=log.WARNING)
    import Modules.Clusters as C

    clusts = C.read_cluster_csv("C:\\Users\\13852\\PycharmProjects\\CMOND\\Datasets\\Vikhlinin.csv")
    for clust in clusts:
        plot_mond_newt_profile(clust)