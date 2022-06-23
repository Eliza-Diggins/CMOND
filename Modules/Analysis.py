"""

         Galaxy Cluster Analysis
        Written by: Eliza Diggins


"""

### IMPORTS ###
import logging as log

import astropy.units as u
import sympy as sym
import numpy as np
import Modules.Clusters as C


### Functions ###
def get_M500(cluster, output_units=u.solMass, gravitational_mode="NEWT"):
    """
    Finds the mass contained within r_500 of the given cluster.
    :param cluster: The cluster object to analyze.
    :return: The mass in the choosen output units.
    """
    # Intro logging
    log.info("CMOND:Analysis:get_M500:INFO: Attempting to compute the r_500 mass of %s." % cluster.name)

    # Checking for sanity inputs
    if cluster.r_500 == None:  # We don't have an r_500 value.
        log.error("CMOND:Analysis:get_M500:ERROR: Failed to find valid r_500 for %s. (%s). Returning False." % (
        cluster.name, cluster.r_500))
        return False
    elif not cluster._has_dynamical_mass_profile_newt and gravitational_mode == "NEWT":  # failed to find the necessary mass profile.
        log.error(
            "CMOND:Analysis:get_M500:ERROR: Failed to find a newtonian mass profile for %s. Returning False." % cluster.name)
        return False
    elif not cluster._has_dynamical_mass_profile_mond and gravitational_mode == "MOND":  # failed to find the necessary mass profile.
        log.error(
            "CMOND:Analysis:get_M500:ERROR: Failed to find a milgromian mass profile for %s. Returning False." % cluster.name)
        return False
    elif gravitational_mode not in ["NEWT", "MOND"]:  # The gravitational mode was not a valid option.
        log.error(
            "CMOND:Analysis:get_M500:ERROR: %s is not a valid gravitational mode. Options are 'NEWT', and 'MOND'" % gravitational_mode)

    # Computing
    if gravitational_mode == "MOND":
        return (cluster.dynamical_mass_profile_mond.subs({sym.Symbol("r"): cluster.r_500.to(
            C._CMOND_base_units["m"]).value}) * cluster._dynamical_mass_profile_mond_units).to(output_units)
    else:
        return (cluster.dynamical_mass_profile_newt.subs({sym.Symbol("r"): cluster.r_500.to(
            C._CMOND_base_units["m"]).value}) * cluster._dynamical_mass_profile_newt_units).to(output_units)


def hernquist_profile(cluster: C.Cluster, mode: str = "Schmidt_Allen", h: u.Quantity = 30 * u.kpc,
                      output_units: u.Unit = u.solMass / (u.kpc ** 3), gravitational_mode: str = "NEWT"):
    """
    Defines the Hernquist profile for the central galaxy density profile.
    :param output_units: The units to output with.
    :param cluster: The cluster to analyze
    :param mode: The mode to use, based off of authored results. Options = ["Schmidt_Allen"].
    :param h: The Hernquist scale length.
    :return: [express,units] or False
    :rtype: list[sym.Expr, u.Unit] or bool
    """
    log.info("CMOND:Analysis:hernquist_profile:INFO: Attempting to produce hernquist profile for %s in mode %s." % (
    cluster.name, mode))

    ### Grabbing correct string expression
    if mode == "Schmidt_Allen":
        """
        
        Using Schmidt Allen Mode. We utilize a BCG mass of 5.3e11 *(M_500/(3e14 M_sol))^0.42 M_sol 
        
        (Schmidt, R. W. & Allen, S. W. 2007, MNRAS, 379, 209)
        """
        log.debug("CMOND:Analysis:hernquist_profile:DEBUG: Mode is %s." % mode)

        express_string = "(M*h)/(2*pi*r*((r+h)**3))"  # Computes the string representing the herquist profile.
        M_herquist = (5.3e11 * u.solMass) * (
                    (get_M500(cluster, gravitational_mode=gravitational_mode).to(u.solMass).value) / (3e14)) ** (
                         0.42)  # Computes the hernquist mass
        comp_units = M_herquist.unit / C._CMOND_base_units[
            "m"] ** 3  # Defining the units we will use for the actual computation.
    else:
        log.error(
            "CMOND:Analysis:hernquist_profile:ERROR: Failed to find a mode matching input: %s. Returning False." % (
                mode))
        return False

    ### Computing the expression

    return [output_units,(comp_units.to(output_units)) * sym.sympify(express_string).subs(
        {sym.Symbol("M"): M_herquist.value, sym.Symbol("h"): h.to(C._CMOND_base_units["m"]).value})]

def get_baryonic_mass_profile(cluster:C.Cluster,bounds=None,gravitational_mode:str="NEWT",indep_unit=u.kpc,dep_unit=u.solMass,n=1000):
    """
    Generates the baryonic mass profile from the BCG profile and the gas density profile. This is largely an approximation.
    :param cluster: The cluster to construct the profile for.
    :param bounds: The bounds on the output array. Inputs should be [lower_bound,upper_bound]. Units are necessary.
    :param gravitational_mode: The gravitational mode to use. Options are "MOND" or "NEWT".
    :param indep_unit: The independent unit (a unit of distance).
    :param dep_unit: The dependent unit (a unit of mass).
    :param n: The number of points to sample in the array. default = 1000
    :return: [r,M(<r)] where r is an array of r values, and M(<r) is the baryonic mass contained within r.
    """
    log.info("CMOND:Analysis:get_baryonic_mass_profile:INFO: Building baryonic mass profile for %s."%cluster.name)

    ### Sanitizing the input ###
    if not(cluster._has_dynamical_mass_profile_newt and cluster._has_dynamical_mass_profile_mond):
        log.error("CMOND:Analysis:get_baryonic_mass_profile:ERROR: Cluster does not have the necessary mass profiles for completion.")
        return False

    ### Building the input array ###
    if not bounds: # The bounds are not manually specified, we will do it.
        bounds = [cluster.r_min.to(indep_unit),cluster.r_500.to(indep_unit)]
    else:
        bounds = [i.to(indep_unit) for i in bounds]

    array = np.linspace(bounds[0],bounds[1],n)

    ### Computing hernquist profile ###
    h_profile = hernquist_profile(cluster,gravitational_mode=gravitational_mode)

    ### Building numerical arrays from density and profile ###
    h_array = sym.lambdify(sym.Symbol('r'),h_profile[1],"numpy")(array.to(C._CMOND_base_units["m"]).value)
    gas_density = sym.lambdify(sym.Symbol('r'),cluster.density_profile,"numpy")(array.to(C._CMOND_base_units["m"]).value)

    # Computing the integral
    density = ((h_array*h_profile[0])+(gas_density*cluster._density_profile_output_units)).to(dep_unit/(indep_unit**3))

    dr = (bounds[1]-bounds[0])/n
    dm = density.value*(4*np.pi*(array.value**2))*dr.value


    return [array,np.array([sum(dm[:i]) for i in range(len(array))])*dep_unit]





if __name__ == '__main__':
    log.basicConfig(level="INFO")

    clusts = C.read_cluster_csv("C:\\Users\\13852\\PycharmProjects\\CMOND\\Datasets\\Vikhlinin.csv")
    for clust in clusts:
        print(clust.name,hernquist_profile(clust),get_baryonic_mass_profile(clust))
