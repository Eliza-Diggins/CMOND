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
import Modules.Analysis as ana
import os
from datetime import datetime
import gc
import matplotlib as mpl

### Functions

def plot_temperature_profiles(clusters,
                              r_min=None,
                              r_max=None,
                              n=500,
                              indp_unit = u.kpc,
                              dep_unit = u.keV,
                              scale="semilogx",save=True,end_file=None
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
        try:
            r_min = min([c.r_min for c in USABLE_CLUSTERS if c.r_min != None])
        except Exception:
            r_min = 1*u.kpc
        CUSTOM_MIN=False
    else:
        CUSTOM_MIN = True
    if r_max == None:
        try:
            r_max = max([c.r_500 for c in USABLE_CLUSTERS if c.r_500 != None])
        except:
            r_max = 1000*u.kpc
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

    if save:
        if len(clusters) == 1:
            if end_file:
                try:
                    os.chdir(end_file)
                except FileNotFoundError:
                    os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                    if os.path.isdir(clusters[0].name):
                        os.chdir(clusters[0].name)
                    else:
                        os.mkdir(clusters[0].name)
                        os.chdir(clusters[0].name)
                plt.savefig("Temperature_Profile_%s.png" % (clusters[0].name))
            else:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir(clusters[0].name):
                    os.chdir(clusters[0].name)
                else:
                    os.mkdir(clusters[0].name)
                    os.chdir(clusters[0].name)
                plt.savefig("Temperature_Profile_%s.png" % (clusters[0].name))
        else:
            if end_file:
                try:
                    os.chdir(end_file)
                except FileNotFoundError:
                    os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                    if os.path.isdir("ensemble"):
                        os.chdir("ensemble")
                    else:
                        os.mkdir("ensemble")
                        os.chdir("ensemble")
                plt.savefig("Temperature_Profile_%s.png" % datetime.now().strftime(
                    '%m-%d-%Y_%H-%M-%S'))
            else:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir("ensemble"):
                    os.chdir("ensemble")
                else:
                    os.mkdir("ensemble")
                    os.chdir("ensemble")
                plt.savefig("Temperature_Profile_%s.png" % datetime.now().strftime(
                    '%m-%d-%Y_%H-%M-%S'))
    else:
        plt.show()

    ### Garbage Collection ###
    plt.figure().clear()
    plt.clf()
    plt.close('all')
    gc.collect()


def plot_density_profiles(clusters,
                              r_min=None,
                              r_max=None,
                              n=500,
                              indp_unit=u.kpc,
                              dep_unit=u.solMass/(u.m**3),
                              scale="semilogx",save=True,end_file=None
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
    if save:
        mpl.use("agg")
    else:
        mpl.use('TkAgg')

    # Cleaning clusters
    USABLE_CLUSTERS = [cluster for cluster in clusters if cluster._has_density_profile]
    log.debug(
        "CMOND:Visualization:plot_temperature_profiles:DEBUG: Usable clusters are %s" % [cluster.name for cluster in
                                                                                         USABLE_CLUSTERS])

    # Fixing bounds
    if r_min == None:
        try:
            r_min = min([c.r_min for c in USABLE_CLUSTERS if c.r_min != None])
        except Exception:
            r_min = 1 * u.kpc
        CUSTOM_MIN = False
    else:
        CUSTOM_MIN = True
    if r_max == None:
        try:
            r_max = max([c.r_500 for c in USABLE_CLUSTERS if c.r_500 != None])
        except:
            r_max = 1000 * u.kpc
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

    if save:
        if len(clusters) == 1:
            if end_file:
                try:
                    os.chdir(end_file)
                except FileNotFoundError:
                    os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                    if os.path.isdir(clusters[0].name):
                        os.chdir(clusters[0].name)
                    else:
                        os.mkdir(clusters[0].name)
                        os.chdir(clusters[0].name)
                plt.savefig("Density_Profile_%s.png" % (clusters[0].name))
            else:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir(clusters[0].name):
                    os.chdir(clusters[0].name)
                else:
                    os.mkdir(clusters[0].name)
                    os.chdir(clusters[0].name)
                plt.savefig("Density_Profile_%s.png" % (clusters[0].name))
        else:
            if end_file:
                try:
                    os.chdir(end_file)
                except FileNotFoundError:
                    os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                    if os.path.isdir("ensemble"):
                        os.chdir("ensemble")
                    else:
                        os.mkdir("ensemble")
                        os.chdir("ensemble")
                plt.savefig("Density_Profile_%s.png" % datetime.now().strftime(
                    '%m-%d-%Y_%H-%M-%S'))
            else:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir("ensemble"):
                    os.chdir("ensemble")
                else:
                    os.mkdir("ensemble")
                    os.chdir("ensemble")
                plt.savefig("Density_Profile_%s.png" % datetime.now().strftime(
                    '%m-%d-%Y_%H-%M-%S'))
    else:
        plt.show()

    ### Garbage Collection ###
    plt.figure().clear()
    plt.clf()
    plt.close('all')
    gc.collect()

def plot_dynamical_mass_profiles(clusters,
                              r_min=None,
                              r_max=None,
                              n=500,
                              indp_unit=u.kpc,
                              dep_unit=u.solMass,
                              scale="semilogx",
                              mode="NEWT",save=True,end_file=None
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
    if save:
        mpl.use("agg")
    else:
        mpl.use('TkAgg')
    # Cleaning clusters
    USABLE_CLUSTERS = [cluster for cluster in clusters if cluster._has_dynamical_mass_profile_newt]
    log.debug(
        "CMOND:Visualization:plot_temperature_profiles:DEBUG: Usable clusters are %s" % [cluster.name for cluster in
                                                                                         USABLE_CLUSTERS])

    # Fixing bounds
    if r_min == None:
        try:
            r_min = min([c.r_min for c in USABLE_CLUSTERS if c.r_min != None])
        except Exception:
            r_min = 1 * u.kpc
        CUSTOM_MIN = False
    else:
        CUSTOM_MIN = True
    if r_max == None:
        try:
            r_max = max([c.r_500 for c in USABLE_CLUSTERS if c.r_500 != None])
        except:
            r_max = 1000 * u.kpc
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

    if save:
        if len(clusters) == 1:
            if end_file:
                try:
                    os.chdir(end_file)
                except FileNotFoundError:
                    os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                    if os.path.isdir(clusters[0].name):
                        os.chdir(clusters[0].name)
                    else:
                        os.mkdir(clusters[0].name)
                        os.chdir(clusters[0].name)
                plt.savefig("Dynamical_Mass_Profile_%s.png" % (clusters[0].name))
            else:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir(clusters[0].name):
                    os.chdir(clusters[0].name)
                else:
                    os.mkdir(clusters[0].name)
                    os.chdir(clusters[0].name)
                plt.savefig("Dynamical_Mass_Profile_%s.png" % (clusters[0].name))
        else:
            if end_file:
                try:
                    os.chdir(end_file)
                except FileNotFoundError:
                    os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                    if os.path.isdir("ensemble"):
                        os.chdir("ensemble")
                    else:
                        os.mkdir("ensemble")
                        os.chdir("ensemble")
                plt.savefig("Dynamical_Mass_Profile_%s.png" % datetime.now().strftime(
                    '%m-%d-%Y_%H-%M-%S'))
            else:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir("ensemble"):
                    os.chdir("ensemble")
                else:
                    os.mkdir("ensemble")
                    os.chdir("ensemble")
                plt.savefig("Dynamical_Mass_Profile_%s.png" % datetime.now().strftime(
                    '%m-%d-%Y_%H-%M-%S'))
    else:
        plt.show()

    ### Garbage Collection ###
    plt.figure().clear()
    plt.clf()
    plt.close('all')
    gc.collect()

def plot_mond_newt_profile(cluster,r_min=None,r_max=None,indp_unit=u.kpc,scale="loglog",dep_unit=u.solMass,n=1000,save=True,end_file=None):
    """
    Plots both the milgromian and newtonian profiles for the mass.
    :param cluster: The cluster to plot
    :return: None
    """
    log.info(
        "CMOND:Visualizations:plot_mond_newt_profile:INFO plotting mond and newt profiles for %s." % cluster.name)

    if save:
        mpl.use("agg")
    else:
        mpl.use('TkAgg')

    if not (cluster._has_dynamical_mass_profile_newt and cluster._has_dynamical_mass_profile_mond):
        raise NotImplementedError("CMOND:Visualizations:plot_mond_newt_profile:ERROR: Failed to find necessary profiles for plotting.")

    # Fixing bounds
    print(r_min,r_max)
    if r_min == None:
        try:
            r_min = cluster.r_min
            if r_min == None:
                raise ValueError
        except Exception:
            r_min = 1*u.kpc
        CUSTOM_MIN = False
    else:
        CUSTOM_MIN = True
    if r_max == None:
        try:
            r_max = cluster.r_500
            if r_max == None:
                raise ValueError
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

    if save:
        if end_file:
            try:
                os.chdir(end_file)
            except FileNotFoundError:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir(cluster.name):
                    os.chdir(cluster.name)
                else:
                    os.mkdir(cluster.name)
                    os.chdir(cluster.name)
            plt.savefig("MOND_NEWT_Profile_%s.png" % (cluster.name))
        else:
            os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
            if os.path.isdir(cluster.name):
                os.chdir(cluster.name)
            else:
                os.mkdir(cluster.name)
                os.chdir(cluster.name)
            plt.savefig("MOND_NEWT_Profile_%s.png" % (cluster.name))
    else:
        plt.show()

    ### Garbage Collection ###
    plt.figure().clear()
    plt.clf()
    plt.close('all')
    gc.collect()

def plot_mass_profile(cluster,mode="Baryonic",bounds=None,indep_unit = u.kpc,dep_unit=u.solMass,scale="loglog",n=1000,gravitational_mode="NEWT",style=None,save=True,end_file=None):
    log.info(
        "CMOND:Visualizations:plot_mass_profile:INFO plotting mass profile for %s." % cluster.name)

    if save:
        mpl.use("agg")
    else:
        mpl.use('TkAgg')

    ### Checking for necessary profiles ###
    if not (cluster._has_dynamical_mass_profile_newt and cluster._has_dynamical_mass_profile_mond):
        log.error(
            "CMOND:Analysis:plot_mass_profile:ERROR: Cluster does not have the necessary mass profiles for completion.")
        return False

    # Fixing bounds
    if not bounds: # No bounds were specified.
        try:
            bounds = [cluster.r_min.to(indep_unit),cluster.r_500.to(indep_unit)]
        except u.UnitConversionError:
            log.error("CMOND:Visualizations:plot_mass_profile:ERROR: Cluster %s could not be plotted because r_min = %s, r_max = %s; one of which is not a unit of length."%(cluster.name,cluster.r_min,cluster.r_500))
            return False
        except AttributeError:
            log.error("CMOND:Visualizations:plot_mass_profile:ERROR: Cluster %s does not have either an r_min or and r_500. Returning False."%cluster.name)
            return False
    if not style:
        style = ['-b','-r','--k']

    # Creating the figure and the axes
    if mode == 'Baryonic' or mode == "Dynamical":
        fig = plt.figure(figsize=(10, 6))
        axes = [fig.add_subplot(111)]
    else:
        fig, axes = plt.subplots(ncols=1,nrows=2,gridspec_kw={"hspace":0,"height_ratios":[3,1]},sharex=True,figsize=(7,9))

    # Managing the scale

    if scale == "semilogx":
        axes[0].set_xscale("log")
    elif scale == "semilogy":
        axes[0].set_yscale("log")
    elif scale == "loglog":
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
    else:
        log.warning("CMOND:Visualizations:plot_mass_profile:WARNING: %s was not a recognized scale option."%scale)
        pass

    ### Plotting ###
    if mode == "Baryonic":
        ### We are plotting just the baryonic mass profile.
        data = ana.get_baryonic_mass_profile(cluster,bounds=bounds,gravitational_mode=gravitational_mode,indep_unit=indep_unit,dep_unit=dep_unit,n=n) # Fetching data

        ### plotting
        X = data[0].value
        Y = data[1].value

        axes[0].plot(X,Y,style[0],label="Bary. $M(<r)$")

        ### Customization ###
        plt.title("Baryonic Mass Profile for %s [%s]."%(cluster.name,gravitational_mode))
        axes[0].set_xlabel("Radius [%s]"%indep_unit)
    elif mode == "Dynamical":
        if gravitational_mode == "MOND":
            # We use the dynamical mass from MOND
            X = np.linspace(bounds[0],bounds[1],n).value
            Y = sym.lambdify(sym.Symbol("r"),cluster.dynamical_mass_profile_mond,"numpy")(X)*(cluster._dynamical_mass_profile_mond_units.to(dep_unit))
        elif gravitational_mode == "NEWT":
            # We use the dynamical mass from MOND
            X = np.linspace(bounds[0],bounds[1],n).value
            Y = sym.lambdify(sym.Symbol("r"),cluster.dynamical_mass_profile_newt,"numpy")(X)*(cluster._dynamical_mass_profile_newt_units.to(dep_unit))
        else:
            log.error("CMOND:Visualizations:plot_mass_profile:ERROR: %s is not a supported gravitational mode. Please use 'MOND' or 'NEWT'"%gravitational_mode)
            return False

        axes[0].plot(X,Y,style[0],label="Dyna. $M(<r)$")

        ### Customization ###
        plt.title("Dynamical Mass Profile for %s [%s]."%(cluster.name,gravitational_mode))
        axes[0].set_xlabel("Radius [%s]"%indep_unit)

    else:
        """
        We are plotting both cases.
        """
        data = ana.get_baryonic_mass_profile(cluster, bounds=bounds, gravitational_mode=gravitational_mode,
                                             indep_unit=indep_unit, dep_unit=dep_unit, n=n)  # Fetching data

        ### plotting
        X = data[0].value
        Y1 = data[1].value

        axes[0].plot(X, Y1,style[0], label="Bary. $M(<r)$")

        if gravitational_mode == "MOND":
            # We use the dynamical mass from MOND
            X = np.linspace(bounds[0], bounds[1], n).value
            Y2 = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_mond, "numpy")(X) * (
                cluster._dynamical_mass_profile_mond_units.to(dep_unit))
        elif gravitational_mode == "NEWT":
            # We use the dynamical mass from MOND
            X = np.linspace(bounds[0], bounds[1], n).value
            Y2 = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_newt, "numpy")(X) * (
                cluster._dynamical_mass_profile_newt_units.to(dep_unit))
        else:
            log.error(
                "CMOND:Visualizations:plot_mass_profile:ERROR: %s is not a supported gravitational mode. Please use 'MOND' or 'NEWT'" % gravitational_mode)
            return False
        axes[0].plot(X, Y2,style[1], label="Dyna. $M(<r)$")

        ### Plotting the residuals
        axes[1].plot(X,abs(Y2-Y1)/np.amax(Y2-Y1),style[2])
        axes[1].set_yscale("linear")
        axes[1].set_ylim([0,1.1])

        ### Customization ###
        axes[0].set_title("Dynamical and Baryonic Mass Profiles for %s [%s]."%(cluster.name,gravitational_mode))
        axes[1].set_ylabel(r"Scaled Residual $\frac{|\Delta M(<r)|}{\Delta M_{max}}$")
        axes[1].set_xlabel("Radius [%s]"%indep_unit)


    ### Mode independent customization ###
    axes[0].set_ylabel(r"Mass Enclosed $M(<r)$ [%s]"%dep_unit)
    for ax in axes:
        ax.grid()

    axes[0].legend()

    ### SAVE
    if save:
        if end_file:
            try:
                os.chdir(end_file)
            except FileNotFoundError:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir(cluster.name):
                    os.chdir(cluster.name)
                else:
                    os.mkdir(cluster.name)
                    os.chdir(cluster.name)
            plt.savefig("MassProfile_%s%%%s.png"%(cluster.name,gravitational_mode))
        else:
            os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
            if os.path.isdir(cluster.name):
                os.chdir(cluster.name)
            else:
                os.mkdir(cluster.name)
                os.chdir(cluster.name)
            plt.savefig("MassProfile_%s%%%s.png" % (cluster.name, gravitational_mode))
    else:
        plt.show()

    plt.figure().clear()
    for i, ax in enumerate(axes):
        ax.cla()
    plt.clf()
    plt.close('all')
    gc.collect()

def plot_residuals(clusters,save=True,bounds=None,end_file=None,gravitational_mode="BOTH",indep_unit=u.kpc,dep_unit=u.solMass,ls=None,c=None,n=1000):
    ### Intro logging
    log.info(
        "CMOND:Visualizations:plot_mass_profile:INFO plotting mass profile for %s." % [c.name for c in clusters])

    ### Setting backend mode.
    if save:
        mpl.use("agg")
    else:
        mpl.use('TkAgg')

    if not ls:
        ls = ["-","-"]

    if not c:
        c= ["r","b"]
    ### Building figure
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)

    for cluster in clusters:
        CHECKER = True
        ### Managing bounds
        if not bounds:  # No bounds were specified.
            try:
                bounds = [cluster.r_min.to(indep_unit), cluster.r_500.to(indep_unit)]
            except u.UnitConversionError:
                log.error(
                    "CMOND:Visualizations:plot_mass_profile:ERROR: Cluster %s could not be plotted because r_min = %s, r_max = %s; one of which is not a unit of length." % (
                    cluster.name, cluster.r_min, cluster.r_500))
                CHECKER = False
            except AttributeError:
                log.error(
                    "CMOND:Visualizations:plot_mass_profile:ERROR: Cluster %s does not have either an r_min or and r_500." % cluster.name)
                CHECKER = False

        if CHECKER:
            try:
                if gravitational_mode == "MOND":
                    # We use the dynamical mass from MOND
                    data = ana.get_baryonic_mass_profile(cluster, bounds=bounds, gravitational_mode="MOND",
                                                         indep_unit=indep_unit, dep_unit=dep_unit, n=n)  # Fetching data

                    X = np.linspace(bounds[0], bounds[1], n).value  # grabbing the dep variable array
                    Y1 = data[1].value

                    Y2 = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_mond, "numpy")(X) * (
                        cluster._dynamical_mass_profile_mond_units.to(dep_unit))
                    ax1.plot(X, abs(Y2 - Y1) / np.amax(Y2 - Y1), linestyle=ls[1],color=c[1])
                elif gravitational_mode == "NEWT":
                    # We use the dynamical mass from NEWT
                    data = ana.get_baryonic_mass_profile(cluster, bounds=bounds, gravitational_mode="NEWT",
                                                         indep_unit=indep_unit, dep_unit=dep_unit, n=n)  # Fetching data

                    X = np.linspace(bounds[0], bounds[1], n).value  # grabbing the dep variable array
                    Y1 = data[1].value

                    Y2 = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_newt, "numpy")(X) * (
                        cluster._dynamical_mass_profile_newt_units.to(dep_unit))
                    ax1.plot(X, abs(Y2 - Y1) / np.amax(Y2 - Y1), linestyle=ls[0],color=c[0])
                elif gravitational_mode == "BOTH":
                    data = ana.get_baryonic_mass_profile(cluster, bounds=bounds, gravitational_mode="NEWT",
                                                         indep_unit=indep_unit, dep_unit=dep_unit, n=n)  # Fetching data

                    X = np.linspace(bounds[0], bounds[1], n).value  # grabbing the dep variable array
                    Y1 = data[1].value
                    Y2 = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_newt, "numpy")(X) * (
                        cluster._dynamical_mass_profile_newt_units.to(dep_unit))

                    NORM = np.amax(Y2-Y1)
                    ax1.plot(X, abs(Y2 - Y1) / NORM, linestyle=ls[0],color=c[0])
                    data = ana.get_baryonic_mass_profile(cluster, bounds=bounds, gravitational_mode="MOND",
                                                         indep_unit=indep_unit, dep_unit=dep_unit, n=n)  # Fetching data

                    X = np.linspace(bounds[0], bounds[1], n).value  # grabbing the dep variable array
                    Y1 = data[1].value
                    Y2 = sym.lambdify(sym.Symbol("r"), cluster.dynamical_mass_profile_mond, "numpy")(X) * (
                        cluster._dynamical_mass_profile_mond_units.to(dep_unit))
                    ax1.plot(X, abs(Y2 - Y1) / NORM,linestyle=ls[1],color=c[1])

                else:
                    log.error(
                        "CMOND:Visualizations:plot_mass_profile:ERROR: %s is not a supported gravitational mode. Please use 'MOND' or 'NEWT'" % gravitational_mode)
                    return False
            except Exception:
                pass
        else:
            pass

    ## Customization
    ax1.legend([mpl.lines.Line2D([],[],linestyle=ls[0],color=c[0]),mpl.lines.Line2D([],[],linestyle=ls[1],color=c[1])],["Newtonian","Milgromian"])
    plt.grid()
    ax1.set_xlabel("Radial Distance [%s]"%indep_unit)
    ax1.set_ylabel("Scaled Missing Mass")
    ax1.set_title("Mass Deficit For %s Clusters"%len(clusters))
    ### SAVE
    if save:
        if len(clusters) == 1:
            if end_file:
                try:
                    os.chdir(end_file)
                except FileNotFoundError:
                    os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                    if os.path.isdir(clusters[0].name):
                        os.chdir(clusters[0].name)
                    else:
                        os.mkdir(clusters[0].name)
                        os.chdir(clusters[0].name)
                plt.savefig("Residual_Mass_Profile_%s.png" % (clusters[0].name))
            else:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir(clusters[0].name):
                    os.chdir(clusters[0].name)
                else:
                    os.mkdir(clusters[0].name)
                    os.chdir(clusters[0].name)
                plt.savefig("Residual_Mass_Profile_%s.png" % (clusters[0].name))
        else:
            if end_file:
                try:
                    os.chdir(end_file)
                except FileNotFoundError:
                    os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                    if os.path.isdir("ensemble"):
                        os.chdir("ensemble")
                    else:
                        os.mkdir("ensemble")
                        os.chdir("ensemble")
                plt.savefig("Residual_Mass_Profile_%s.png" % datetime.now().strftime(
                    '%m-%d-%Y_%H-%M-%S'))
            else:
                os.chdir("C:\\Users\\13852\\PycharmProjects\\CMOND\\Graphics")
                if os.path.isdir("ensemble"):
                    os.chdir("ensemble")
                else:
                    os.mkdir("ensemble")
                    os.chdir("ensemble")
                plt.savefig("Residual_Mass_Profile_%s.png" % datetime.now().strftime(
                    '%m-%d-%Y_%H-%M-%S'))
    else:
        plt.show()

    ### Garbage Collection ###
    plt.figure().clear()
    plt.clf()
    plt.close('all')
    gc.collect()


if __name__ == '__main__':
    log.basicConfig(level=log.DEBUG)
    import Modules.Clusters as C

    clusts = C.read_cluster_csv("C:\\Users\\13852\\PycharmProjects\\CMOND\\Datasets\\Vikhlinin.csv")
    plot_residuals(clusts)
"""
    clusts = C.read_cluster_csv("C:\\Users\\13852\\PycharmProjects\\CMOND\\Datasets\\Vikhlinin.csv")
    for clust in clusts:
        plot_mond_newt_profile(clust)
        plot_density_profiles([clust])
        plot_temperature_profiles([clust])
        plot_dynamical_mass_profiles([clust])
        plot_mass_profile(clust,mode="BOTH")

    plot_density_profiles(clusts)
    plot_temperature_profiles(clusts)
    plot_dynamical_mass_profiles(clusts)"""