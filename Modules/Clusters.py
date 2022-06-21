"""

    Galaxy Cluster Models and Analysis
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

######################################
#                                    #
#          Unit Management           #
#                                    #
######################################
_CMOND_base_units = {"m":u.kpc,
                     "kg":u.solMass,
                     "K":u.K,
                     "s":u.s,
                     "E":u.keV}
######################################
#                                    #
#          State Variables           #
#                                    #
######################################
_mu = 1.624  # The constant used to convert n_pn_e(r) to gas mass.
_nu = 0.5954 # The metalicity parameter of the universe.
_a0 = 1.2e-11 *(u.m/u.s**2) # The MOND acceleration constant.

######################################
#                                    #
#         Class Definitions          #
#                                    #
######################################

def standard_interpolation_function(x):
    return x/(1+x)

class Cluster():
    def __init__(self,name=None,sphS=True):
        """
        Defines a given cluster with a choice of model.
        :param name: The name of the cluster. None will produce a random cluster name.
        :param sphS: True if the cluster is to be treated as spherically symmetric. False to use a more complex coordinate system. TODO: implementation.
        """

        # Intro debugging
        log.debug("CMOND:Clusters:Cluster:DEBUG: Creating a cluster with name <%s>"%name)

        # Naming the cluster

        if name:
            self.name = name
        else: # Generating random name
            self.name = "CL"+str(np.random.randint(0,99999))

        # Adding key namespace parameters
        self.r_min = None
        self.r_500 = None
        self._has_dynamical_mass_profile_newt = False
        self._has_temp_profile = False
        self._has_density_profile=False


    def temp_fit_sym(self,params,model):
        """
        Creates a temperature fit as a function of radius for spherically symetric clusters
        :param params: The parameters as a dictionary of values with astropy units.
        :type params: dict[str,u.quantity.Quantity or float]
        :param model: The type of the temperature fit. Options: ["VIKH"]
        :type model: str
        :return: sympy expression of the fit with the correct values of the parameters.
        """
        ### Handling params units ###

        new_params = {param: convert_to_base_units(params[param]).value for param in params}
        unit_params = {param: convert_to_base_units(params[param]).unit for param in params}
        log.debug("CMOND:Clusters:Cluster:temp_fit:INFO: Converted params %s to %s in base units." % (
            params, new_params))

        if model == "VIKH":
            """
            Using the Vihklinin 2006 paper fits for temperature: namely
            
            T(r) = T_0[((r/r_cool)+(T_min/T_0))/((r/r_cool)^a_cool + 1)] ((r/r_t)^-a/((r/r_t)^b+1)^(c/b))
            """
            # Defining the base expression
            expression_string_literal = "T_0*(((r/r_cool)**a_cool+(T_min/T_0))/((r/r_cool)**a_cool + 1))*((r/r_t)**(-a)/((r/r_t)**b+1)**(c/b))"

            # Converting to symbolic expression
            expression = sym.sympify(expression_string_literal)


            self.temp_profile = expression.subs([(sym.symbols(i),new_params[i]) for i in new_params]) # substitutes all of the parameters
            self._has_temp_profile = True
            self._temp_profile_output_units = unit_params["T_0"]

    def density_fit_sym(self, params, model):
        """
        Creates a density fit as a function of radius for spherically symetric clusters
        :param params: The parameters as a dictionary of values with astropy units.
        :type params: dict[str,u.quantity.Quantity or float]
        :param model: The type of the temperature fit. Options: ["VIKH"]
        :type model: str
        :return: sympy expression of the fit with the correct values of the parameters.
        """
        ### Handling params units ###
        # We are working in united values
        new_params = {param: convert_to_base_units(params[param]).value for param in params}
        unit_params = {param: convert_to_base_units(params[param]).unit for param in params}
        log.debug("CMOND:Clusters:Cluster:temp_fit:INFO: Converted params %s to %s in base units." % (
            params, new_params))

        if model == "VIKH":
            """
            Using the Vihklinin 2006 paper fits for temperature: namely

            n_en_p (r) = (n_0**2)*(((r/r_c)**(-a))/((1+(r/r_c)**2)**(3*beta-(a/2))))*(1/((1+(r/r_s))**(epsilon/gamma))) + ((n_0**2)/((1+(r/r_c2)**2)**(3*beta_2)))
            """
            # Defining the base expression
            expression_string_literal =  '%s*m_p*((n_0**2)*(((r/r_c)**(-ALPHA))/((1+(r/r_c)**2)**(3*BETA-(ALPHA/2))))*(1/((1+(r/r_s)**(GAMMA))**(EPSILON/GAMMA))) + ((n_02**2)/((1+(r/r_c2)**2)**(3*BETA_2))))**(1/2)'%_mu

            # Converting to symbolic expression
            expression = sym.sympify(expression_string_literal)


            self.density_profile = expression.subs(
                [(sym.symbols(i), new_params[i]) for i in new_params])  # substitutes all of the parameters
            self._has_density_profile = True

            self._density_profile_output_units = (unit_params["m_p"]*unit_params["n_0"])

    def dynamical_mass_fit_sym(self,gravitational_mode="NEWT",interpolation_function="x/(x+1)",nu=0.5954):
        """
        Constructs the dynamical mass of the system between the two given radii
        :param gravitational_mode:
        :param interpolation_function:
        :return:
        """
        # Intro logging
        log.debug("CMOND:Clusters:Cluster:dynamical_mass_fit_sym:INFO: Fitting mass to cluster %s."%self.name)

        # Sanitizing
        if not self._has_density_profile or not self._has_temp_profile: # The required data are not there.
            raise NotImplementedError("CMOND:Clusters:Cluster:dynamical_mass_fit_sym:ERROR: Cluster %s does not have sufficient models for this computation."%self.name)


        # Creating the radial array
        """
        Completing the necessary computations to find the dynamical mass.
            
            Note: In the newtonian formalism,
            
            M(<r) = -k*r^2/G*nu*m_p [drho_g/dr*(T/rho) + dT/dr]
        
        """
        ### Compute the derivatives
        dTdr_sym = sym.diff(self.temp_profile,sym.Symbol("r"))
        drhodr_sym = sym.diff(self.density_profile,sym.Symbol("r"))

        ### Computing the coefficient
        coeff = 1/(consts.G*nu*consts.m_p)
        coeff_mond = 1/(consts.m_p*nu*_a0)

        ### Handling units
        output_units = self._temp_profile_output_units*_CMOND_base_units["m"]*coeff.unit

        ### Managing modes
        if gravitational_mode == "NEWT":
            ### Symbolic mass function
            self.dynamical_mass_profile_newt = -(sym.Symbol("r")**2)*coeff.value*(((self.temp_profile/self.density_profile)*(drhodr_sym))+(dTdr_sym))
            self._has_dynamical_mass_profile_newt = True
            self._dynamical_mass_profile_newt_units = output_units
        if gravitational_mode == "MOND":
            ### Symbolic mass function
            self.dynamical_mass_profile_mond =-(sym.Symbol("r")**2)*coeff.value*(((self.temp_profile/self.density_profile)*(drhodr_sym))+(dTdr_sym))*(( sym.sympify(interpolation_function).subs({sym.Symbol("x"):abs(coeff_mond.to(_CMOND_base_units["m"]/self._temp_profile_output_units).value*(((self.temp_profile/self.density_profile)*(drhodr_sym))+(dTdr_sym)))})))
            self._has_dynamical_mass_profile_mond = True
            self._dynamical_mass_profile_mond_units = output_units




######################################
#                                    #
#       Function Definitions         #
#                                    #
######################################

def convert_to_base_units(quantity:u.quantity.Quantity) -> u.quantity.Quantity:
    """
    Converts the input quantity to the units for the library.

    Units are assigned by setting values in _CMOND_base_units variable
    :param quantity: The quantity to convert to base units.
    :return: Returns the converted value of the quantity.
    """
    log.debug("CMOND:Clusters:convert_to_base_units:DEBUG: Attempting to convert %s to base units."%quantity)
    # getting string representation in base units
    unit_string = str(quantity.decompose(bases=u.si.bases).unit) # Grabs the units of the quantity and creates a string

    ### MANUAL EXCEPTIONS ###
    if "kg m2 / s2" in unit_string:
        ### The units are of energy, we convert to the energy unit
        return quantity.to(_CMOND_base_units["E"])
    # Unit conversion
    for base_unit in _CMOND_base_units: # Iterate through each base unit
        if base_unit in unit_string: # the unit is involved and needs to be replaced
            unit_string = unit_string.replace(base_unit,str(_CMOND_base_units[base_unit]))
            log.debug("CMOND:Clusters:convert_to_base_units:DEBUG: Unit string of %s is now %s."%(quantity,unit_string))

    log.debug("CMOND:Clusters:convert_to_base_units:INFO: Converted %s to %s"%(quantity,quantity.to(u.Unit(unit_string))))
    return quantity.to(u.Unit(unit_string)) # returning the correct quantity after conversion of the string.

def read_cluster_csv(filename:str,mode:str ="VIKH"):
    """
    Reads excel file with the given file name and generates clusters for each one.
    :param filename: The file path to the given data set.
    :param mode: The mode to use for the fits.
    :return: Returns the list of generated clusters
    """
    ## Intro logging
    log.info("CMOND:Clusters:read_cluster_csv:INFO: Attempting to construct clusters from %s."%filename)

    ## Reading the file
    if os.path.isfile(filename):
        # The file exists
        try:
            dataframe = pd.read_csv(filename)
        except Exception:
            raise ReferenceError("CMOND:Clusters:read_cluster_csv:ERROR: Failed to read %s as a .csv."%filename)
    else:
        raise FileNotFoundError("CMOND:Clusters:read_cluster_csv:ERROR: Failed to find %s."%filename)


    ### BUILDING THE CLUSTERS ###

    CLUSTERS = list(dataframe["Cluster"]) # Reads the column of cluster names.

    output_clusters = []
    for cluster in CLUSTERS:
        temp_cluster = Cluster(name=cluster) # Creates the cluster with the correct name

        ### Building models
        params = {i.split(",")[0]:dataframe.loc[dataframe["Cluster"]==cluster,i].item()*u.Unit(i.split(",")[1][1:-1]) for i in dataframe.columns[1:]}

        # Building the temperature distribution
        if mode=="VIKH":
            # We are running in mode with Vikhlinin et al. 2006
            try:
                temp_cluster.density_fit_sym(params,"VIKH")

            except Exception:
                log.warning("CMOND:Clusters:read_cluster_CSV:WARNING: Failed to fit density model from mode: %s to the parameters for cluster %s."%(mode,cluster))
            try:
                temp_cluster.temp_fit_sym(params,"VIKH")
            except Exception:
                log.warning("CMOND:Clusters:read_cluster_CSV:WARNING: Failed to fit temperature model from mode: %s to the parameters for cluster %s."%(mode,cluster))
            try:
                temp_cluster.dynamical_mass_fit_sym(nu=_nu)
            except Exception:
                log.warning("CMOND:Clusters:read_cluster_CSV:WARNING: Failed to fit dynamical mass model from mode: %s to the parameters for cluster %s."%(mode,cluster))

            try:
                temp_cluster.dynamical_mass_fit_sym(nu=_nu,gravitational_mode="MOND")
            except Exception:
                log.warning("CMOND:Clusters:read_cluster_CSV:WARNING: Failed to fit dynamical (mond) mass model from mode: %s to the parameters for cluster %s."%(mode,cluster))

            ### Adding other useful parameters
            if "r_min" in params and not np.isnan(params["r_min"].value):
                temp_cluster.r_min = params["r_min"]
            if "r_500" in params and not np.isnan(params["r_500"].value):
                temp_cluster.r_500 = params["r_500"]


        ### Cleanup
        if temp_cluster._has_temp_profile:
            if temp_cluster.temp_profile == sym.nan:
                temp_cluster.temp_profile = None
                temp_cluster._has_temp_profile = False
        if temp_cluster._has_density_profile:
            if temp_cluster.density_profile == sym.nan:
                temp_cluster._has_density_profile = False
                temp_cluster.density_profile = None


        output_clusters.append(temp_cluster)
    return output_clusters


if __name__ == '__main__':
    log.basicConfig(level=log.WARNING)
    clusters = read_cluster_csv("C:\\Users\\13852\\PycharmProjects\\CMOND\\Datasets\\Vikhlinin.csv")
    for c in clusters:
        c.dynamical_mass_fit_sym()