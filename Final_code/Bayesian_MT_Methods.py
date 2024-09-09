#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import scipy.stats as stats
import numpy
from scipy.stats import poisson, dirichlet
import random


# In[2]:


import Bayesian_MT_Analysis_Input_Data


# In[3]:


class Bayes_MT_Methods_and_Components: 
    def __init__(self, Clean_Data, n_chains = 5): 
        self.Data_Clean = Clean_Data
        self.Compositional_Features = ['K$_{2}$O', 'CaO', 
                                       'TiO$_{2}$', 'SiO$_{2}$',
                                      'Al$_{2}$O$_{3}$', 'Na$_{2}$O',
                                       'MgO', 'FeO', 'MnO', 'Cl', 
                                          'SO$_{3}$', 'P$_{2}$O$_{5}$']
        self.component_dictionary = self.Component_Dictionary()
        self.n_chains = n_chains
        self.minimum_components = 3
        self.maximum_components = 30

    # Centered Log Transform
    def clr_transform(self, Y):
        """
        This function perform a centered log-ratio 
        transform on the dataset
        """
        clr_ = np.log(Y/np.mean(Y, axis = 1)[:,np.newaxis])

        return clr_

    def Log_Prior_K(self, K): 
        """
        log Poisson prior for the number of components K.
        
        Parameters
        ----------
        K : int
            Number of components.

        Returns
        -------
        Log Prior Value for given number of components
        """        
        # Poisson prior part
        log_prior_K_ = poisson.logpmf(K, 20)
        
        return log_prior_K_


    def Jacobian_Birth(self, K, K_max): 
        return np.log(p_birth / (K_max - K))

    def Jacobian_Death(self, K): 
        return np.log(p_death / K)   


    def generate_Granitoid_component(self): 
        # Generating random values with given mean and standard deviation, taking absolute to avoid negative values
        values = [
            abs(np.random.normal(4.3, 2.6)),   # K$_{2}$O
            abs(np.random.normal(1.46, 2.52)), # CaO
            abs(np.random.normal(0.29, 0.25)), # TiO$_{2}$
            abs(np.random.normal(72.16, 8.05)), # SiO$_{2}$
            abs(np.random.normal(13.9, 3)),  # Al$_{2}$O$_{3}$
            abs(np.random.normal(3.6, 2)),  # Na$_{2}$O
            abs(np.random.normal(0.63, 0.1)),  # MgO
            abs(np.random.normal(1.4, 2)),   # FeO
            abs(np.random.normal(0.05, 0.01)), # MnO
            abs(np.random.normal(0.011, 0.03)),# Cl
            abs(np.random.normal(0.05, 0.42)), # SO$_{3}$
            abs(np.random.normal(0.11, 0.33)) # P$_{2}$O$_{5}$
        ]
    
        # Normalize the values
        normalized_values = np.array(values) / np.array(values).sum()
        
        # Create dictionary of component labels and their normalized values
        granitoid_dict = dict(zip(self.Compositional_Features, normalized_values))
        
        return granitoid_dict




    def generate_limestone_component(self):
        # Generating random values with given mean and standard deviation, taking absolute to avoid negative values
        values = [
            abs(np.random.normal(0, 1e-6)),   # K$_{2}$O
            abs(np.random.normal(100, 3)), # CaO
            abs(np.random.normal(0, 1e-6)), # TiO$_{2}$
            abs(np.random.normal(0, 1e-6)), # SiO$_{2}$
            abs(np.random.normal(0, 1e-6)),  # Al$_{2}$O$_{3}$
            abs(np.random.normal(0, 1e-6)),  # Na$_{2}$O
            abs(np.random.normal(0, 1e-6)),  # MgO
            abs(np.random.normal(0, 1e-6)),   # FeO
            abs(np.random.normal(0, 1e-6)), # MnO
            abs(np.random.normal(0, 1e-6)),# Cl
            abs(np.random.normal(0, 1e-6)), # SO$_{3}$
            abs(np.random.normal(0, 1e-6)) # P$_{2}$O$_{5}$
        ]
    
        # Normalize the values
        normalized_values = np.array(values) / np.array(values).sum()
        
        # Create dictionary of component labels and their normalized values
        limestone_dict = dict(zip(self.Compositional_Features, normalized_values))
        
        return limestone_dict
    




    def generate_Shale_component(self): 
            # Generating random values with given mean and standard deviation, taking absolute to avoid negative values
        values = [
            abs(np.random.normal(4, 6)),   # K$_{2}$O
            abs(np.random.normal(2.4, 1.5)), # CaO
            abs(np.random.normal(0.9, 1)), # TiO$_{2}$
            abs(np.random.normal(60, 5)), # SiO$_{2}$
            abs(np.random.normal(18.5, 6)),  # Al$_{2}$O$_{3}$
            abs(np.random.normal(1.8, 0.2)),  # Na$_{2}$O
            abs(np.random.normal(2.9, 0.5)),  # MgO
            abs(np.random.normal(7, 8)),   # FeO
            abs(np.random.normal(0.1, 0.1)), # MnO
            abs(np.random.normal(0.001, 0.001)),# Cl
            abs(np.random.normal(0.0501, 0.001)), # SO$_{3}$
            abs(np.random.normal(0.2, 0.2)) # P$_{2}$O$_{5}$
        ]
    
        # Normalize the values
        normalized_values = np.array(values) / np.array(values).sum()
        
        # Create dictionary of component labels and their normalized values
        shale_dict = dict(zip(self.Compositional_Features, normalized_values))
        
        return shale_dict       


    def generate_Dolomite_component(self):     
        # Generating random values with given mean and standard deviation, taking absolute to avoid negative values
        values = [
            abs(np.random.normal(0, 1e-6)),   # K$_{2}$O
            abs(np.random.normal(50.95, 10)), # CaO
            abs(np.random.normal(0, 1e-6)), # TiO$_{2}$
            abs(np.random.normal(0.64, 1.3)), # SiO$_{2}$
            abs(np.random.normal(0, 1e-6)),  # Al$_{2}$O$_{3}$
            abs(np.random.normal(0, 1e-6)),  # Na$_{2}$O
            abs(np.random.normal(46.97, 10)),  # MgO
            abs(np.random.normal(1.12, 2.12)),   # FeO
            abs(np.random.normal(0, 1e-6)), # MnO
            abs(np.random.normal(0, 1e-6)),# Cl
            abs(np.random.normal(0, 1e-6)), # SO$_{3}$
            abs(np.random.normal(0, 1e-6)) # P$_{2}$O$_{5}$
        ]
    
        # Normalize the values
        normalized_values = np.array(values) / np.array(values).sum()
        
        # Create dictionary of component labels and their normalized values
        dolomite_dict = dict(zip(self.Compositional_Features, normalized_values))
        
        return dolomite_dict

    def generate_Anhydrite_component(self): 
        # Generating random values with given mean and standard deviation, taking absolute to avoid negative values
        values = [
            abs(np.random.normal(0, 1e-6)),   # K$_{2}$O
            abs(np.random.normal(40.14, 10)), # CaO
            abs(np.random.normal(0, 1e-6)), # TiO$_{2}$
            abs(np.random.normal(0, 1e-6)), # SiO$_{2}$
            abs(np.random.normal(0, 1e-6)),  # Al$_{2}$O$_{3}$
            abs(np.random.normal(0, 1e-6)),  # Na$_{2}$O
            abs(np.random.normal(0, 1e-6)),  # MgO
            abs(np.random.normal(0, 1e-6)),   # FeO
            abs(np.random.normal(0, 1e-6)), # MnO
            abs(np.random.normal(0, 1e-6)),# Cl
            abs(np.random.normal(51, 10)), # SO$_{3}$
            abs(np.random.normal(0, 1e-6)) # P$_{2}$O$_{5}$
        ]
    
        # Normalize the values
        normalized_values = np.array(values) / np.array(values).sum()
        
        # Create dictionary of component labels and their normalized values
        anhydrite_dict = dict(zip(self.Compositional_Features, normalized_values))
    
        return anhydrite_dict


    def generate_Halite_component(self): 
    # Generating random values with given mean and standard deviation, taking absolute to avoid negative values
        values = [
            abs(np.random.normal(0, 1e-6)),   # K$_{2}$O
            abs(np.random.normal(0, 1e-6)), # CaO
            abs(np.random.normal(0, 1e-6)), # TiO$_{2}$
            abs(np.random.normal(0, 1e-6)), # SiO$_{2}$
            abs(np.random.normal(0, 1e-6)),  # Al$_{2}$O$_{3}$
            abs(np.random.uniform(40, 60)),  # Na$_{2}$O
            abs(np.random.normal(0, 1e-6)),  # MgO
            abs(np.random.normal(0, 1e-6)),   # FeO
            abs(np.random.normal(0, 1e-6)), # MnO
            abs(np.random.uniform(50, 70)),# Cl
            abs(np.random.normal(0, 1e-6)), # SO$_{3}$
            abs(np.random.normal(0, 1e-6)) # P$_{2}$O$_{5}$
        ]
    
        # Normalize the values
        normalized_values = np.array(values) / np.array(values).sum()
        
        # Create dictionary of component labels and their normalized values
        halite_dict = dict(zip(self.Compositional_Features, normalized_values))
    
        return halite_dict  


    def generate_Mafic_component(self): 

        # Generating random values with given mean and standard deviation, taking absolute to avoid negative values
        values = [
        abs(np.random.normal(1, 1.5)),   # K$_{2}$O
        abs(np.random.normal(2.3, 4)), # CaO
        abs(np.random.normal(1.1, 3)), # TiO$_{2}$
        abs(np.random.normal(43.48, 8)), # SiO$_{2}$
        abs(np.random.normal(10, 2.8)),  # Al$_{2}$O$_{3}$
        abs(np.random.normal(0.6, 0.94)),  # Na$_{2}$O
        abs(np.random.normal(11, 8)),  # MgO
        abs(np.random.normal(13, 9)),   # FeO
        abs(np.random.normal(0.11, 0.2)), # MnO
        abs(np.random.normal(0.225, 0.225)),# Cl
        abs(np.random.normal(0.14, 0.2)), # SO$_{3}$
        abs(np.random.normal(0.11, 0.15)) # P$_{2}$O$_{5}$
    ]

        # Normalize the values
        normalized_values = np.array(values) / np.array(values).sum()
        
        # Create dictionary of component labels and their normalized values
        mafic_dict = dict(zip(self.Compositional_Features, normalized_values))
    
        return mafic_dict

    def generate_Random_component(self): 
            # Generating random values with given mean and standard deviation, taking absolute to avoid negative values
        values = [
            np.random.uniform(0, 0.2),   # K$_{2}$O
            np.random.uniform(0, 0.2), # CaO
            np.random.uniform(0, 0.2), # TiO$_{2}$
            np.random.uniform(0, 0.9), # SiO$_{2}$
            np.random.uniform(0, 0.3),  # Al$_{2}$O$_{3}$
            np.random.uniform(0, 0.5),  # Na$_{2}$O
            np.random.uniform(0, 0.1),  # MgO
            np.random.uniform(0, 0.2),   # FeO
            np.random.uniform(0, 0.1), # MnO
            np.random.uniform(0, 0.5),# Cl
            np.random.uniform(0, 0.05), # SO$_{3}$
            np.random.uniform(0, 0.05) # P$_{2}$O$_{5}$
        ]
        
        # Normalize the values
        normalized_values = np.array(values) / np.array(values).sum()
        
        # Create dictionary of component labels and their normalized values
        random_dict = dict(zip(self.Compositional_Features, normalized_values))
    
        return random_dict

    def Component_Dictionary(self): 
        self.component_dictionary = {
                "Granitoid" : self.generate_Granitoid_component(), 
                "Shale" : self.generate_Shale_component(), 
                "Mafic" : self.generate_Mafic_component(), 
                "Halite" : self.generate_Halite_component(), 
                "Limestone" : self.generate_limestone_component(), 
                "Anhydrite" : self.generate_Anhydrite_component(), 
                "Dolomite" : self.generate_Dolomite_component(), 
                "Random" : self.generate_Random_component()
            
        }
        return self.component_dictionary
        
    def Ensure_component_dictionary(self): 
        if self.component_dictionary is None: 
            self.component_dictionary = self.Component_Dictionary()


    def randomly_select_components(self, n_samples = 20):
        if self.component_dictionary is None: 
            self.Ensure_component_dictionary()
        
        keys = list(self.component_dictionary.keys())
            # Randomly select keys, allowing for the same key to be selected multiple times
        selected_keys = random.choices(keys, k=n_samples)
    
        # Create a list to hold dictionaries for each selected component
        component_list = []
        for key in selected_keys:
            # Create a new dictionary for each selected component
            component_list.append({key: self.component_dictionary[key]})
    
        return component_list


    """
    Constraints
    """
    def Anhydrite_Constraint(self, comp_dict): 
        constraints = {
        'K$_{2}$O': (0, 0.005),
        'CaO': (0.3, 0.7),
        'TiO$_{2}$': (0, 0.005),
        'SiO$_{2}$': (0, 0.005),
        'Al$_{2}$O$_{3}$': (0, 0.005),
        'Na$_{2}$O': (0, 0.005),
        'MgO': (0, 0.005),
        'FeO': (0, 0.005),
        'MnO': (0, 0.005),
        'Cl': (0, 0.005),
        'SO$_{3}$': (0.3, 0.7),
        'P$_{2}$O$_{5}$': (0, 0.005)
        }
    
            # Check each component against its constraints
        for component, (min_val, max_val) in constraints.items():
            # Check if the component exists in the dictionary and is within bounds
            if component not in comp_dict or not (min_val <= comp_dict[component] <= max_val):
                return False  # Return False if any component is out of bounds
    
        return True  # Return True if all components are within bounds   

    def Halite_Constraint(self, comp_dict): 
        constraints = {
        'K$_{2}$O': (0, 0.005),
        'CaO': (0, 0.005),
        'TiO$_{2}$': (0, 0.005),
        'SiO$_{2}$': (0, 0.005),
        'Al$_{2}$O$_{3}$': (0, 0.005),
        'Na$_{2}$O': (0.4, 0.80),
        'MgO': (0, 0.005),
        'FeO': (0, 0.005),
        'MnO': (0, 0.005),
        'Cl': (0.4, 0.8),
        'SO$_{3}$': (0, 0.005),
        'P$_{2}$O$_{5}$': (0, 0.005)
        }
    
            # Check each component against its constraints
        for component, (min_val, max_val) in constraints.items():
            # Check if the component exists in the dictionary and is within bounds
            if component not in comp_dict or not (min_val <= comp_dict[component] <= max_val):
                return False  # Return False if any component is out of bounds
    
        return True  # Return True if all components are within bounds


    def Mafic_Constraint(self, comp_dict): 
        constraints = {
        'K$_{2}$O': (0, 0.1),
        'CaO': (0, 0.4),
        'TiO$_{2}$': (0, 0.05),
        'SiO$_{2}$': (0.05, 0.7),
        'Al$_{2}$O$_{3}$': (0, 0.34),
        'Na$_{2}$O': (0, 0.1),
        'MgO': (0, 0.4),
        'FeO': (0.003, 0.3),
        'MnO': (0, 0.05),
        'Cl': (0.0007, 0.05),
        'SO$_{3}$': (0, 0.05),
        'P$_{2}$O$_{5}$': (0, 0.08)
        }
    
            # Check each component against its constraints
        for component, (min_val, max_val) in constraints.items():
            # Check if the component exists in the dictionary and is within bounds
            if component not in comp_dict or not (min_val <= comp_dict[component] <= max_val):
                return False  # Return False if any component is out of bounds
    
        return True  # Return True if all components are within bounds

    def Dolomite_Constraint(self, comp_dict): 
        constraints = {
        'K$_{2}$O': (0, 0.005),
        'CaO': (0.3, 0.7),
        'TiO$_{2}$': (0, 0.005),
        'SiO$_{2}$': (0, 0.01),
        'Al$_{2}$O$_{3}$': (0, 0.005),
        'Na$_{2}$O': (0, 0.005),
        'MgO': (0.3, 0.7),
        'FeO': (0, 0.06),
        'MnO': (0, 0.005),
        'Cl': (0, 0.005),
        'SO$_{3}$': (0, 0.005),
        'P$_{2}$O$_{5}$': (0, 0.005)
        }
    
            # Check each component against its constraints
        for component, (min_val, max_val) in constraints.items():
            # Check if the component exists in the dictionary and is within bounds
            if component not in comp_dict or not (min_val <= comp_dict[component] <= max_val):
                return False  # Return False if any component is out of bounds
    
        return True  # Return True if all components are within bounds       


    def Shale_Constraint(self, comp_dict): 
        constraints = {
        'K$_{2}$O': (0, 0.2),
        'CaO': (0, 0.1),
        'TiO$_{2}$': (0, 0.1),
        'SiO$_{2}$': (0, 0.8),
        'Al$_{2}$O$_{3}$': (0, 0.25),
        'Na$_{2}$O': (0, 0.05),
        'MgO': (0, 0.1),
        'FeO': (0, 0.2),
        'MnO': (0, 0.1),
        'Cl': (0, 0.05),
        'SO$_{3}$': (0, 0.05),
        'P$_{2}$O$_{5}$': (0, 0.05)
        }
    
            # Check each component against its constraints
        for component, (min_val, max_val) in constraints.items():
            # Check if the component exists in the dictionary and is within bounds
            if component not in comp_dict or not (min_val <= comp_dict[component] <= max_val):
                return False  # Return False if any component is out of bounds
    
        return True  # Return True if all components are within bounds

    def Limestone_Constraint(self, comp_dict): 
        constraints = {
        'K$_{2}$O': (0, 0.005),
        'CaO': (0.85, 1.000),
        'TiO$_{2}$': (0, 0.005),
        'SiO$_{2}$': (0, 0.005),
        'Al$_{2}$O$_{3}$': (0, 0.005),
        'Na$_{2}$O': (0, 0.005),
        'MgO': (0, 0.005),
        'FeO': (0, 0.005),
        'MnO': (0, 0.005),
        'Cl': (0, 0.005),
        'SO$_{3}$': (0, 0.005),
        'P$_{2}$O$_{5}$': (0, 0.005)
        }
    
            # Check each component against its constraints
        for component, (min_val, max_val) in constraints.items():
            # Check if the component exists in the dictionary and is within bounds
            if component not in comp_dict or not (min_val <= comp_dict[component] <= max_val):
                return False  # Return False if any component is out of bounds
    
        return True  # Return True if all components are within bounds       
                    
    def Grainitoid_Constraint(self, comp_dict):
        constraints = {
        'K$_{2}$O': (0.0, 0.2),
        'CaO': (0, 0.3),
        'TiO$_{2}$': (0, 0.8),
        'SiO$_{2}$': (0.3, 0.95),
        'Al$_{2}$O$_{3}$': (0, 0.27),
        'Na$_{2}$O': (0, 0.12),
        'MgO': (0.0, 0.57),
        'FeO': (0, 0.15),
        'MnO': (0, 0.05),
        'Cl': (0.0, 0.05),
        'SO$_{3}$': (0, 0.05),
        'P$_{2}$O$_{5}$': (0, 0.05)
        }
    
            # Check each component against its constraints
        for component, (min_val, max_val) in constraints.items():
            # Check if the component exists in the dictionary and is within bounds
            if component not in comp_dict or not (min_val <= comp_dict[component] <= max_val):
                return False  # Return False if any component is out of bounds
    
        return True  # Return True if all components are within bounds    

    def Random_Constraint(self, comp_dict): 
        constraints = {
        'K$_{2}$O': (0, 0.2),
        'CaO': (0, 0.2),
        'TiO$_{2}$': (0, 0.2),
        'SiO$_{2}$': (0, 1),
        'Al$_{2}$O$_{3}$': (0, 0.3),
        'Na$_{2}$O': (0, 0.1),
        'MgO': (0, 0.2),
        'FeO': (0, 0.2),
        'MnO': (0, 0.2),
        'Cl': (0.0, 0.2),
        'SO$_{3}$': (0, 0.2),
        'P$_{2}$O$_{5}$': (0, 0.2)
        }
    
            # Check each component against its constraints
        for component, (min_val, max_val) in constraints.items():
            # Check if the component exists in the dictionary and is within bounds
            if component not in comp_dict or not (min_val <= comp_dict[component] <= max_val):
                return False  # Return False if any component is out of bounds
     
        return True  # Return True if all components are within bounds

    def check_all_constraints(self, components):
        """
        Check if all components meet their respective constraints.
        
        Parameters:
        components (list): A list of component dictionaries.
        
        Returns:
        bool: True if all components meet their constraints, False otherwise.
        """
        for component in components:
            component_type = list(component.keys())[0]
            feature_dict = component[component_type]
            
            if not check_component_constraints(component_type, feature_dict):
                print(component_type)
                return False  # Return False if any component does not meet its constraints
    
        return True  # Return True if all components meet their constraints


    def component_constraints(self, select_comp, component_types): 
        """
        Check constraints for a list of components based on their types.
        
        Parameters:
        select_comp (list): A list of component dictionaries.
        component_types (list): A list of strings representing the type of each component.
        
        Returns:
        bool: True if all components meet their constraints, False otherwise.
        """
        for component, comp_type in zip(select_comp, component_types):
            # Get the type of the current component
            component_type = list(component.keys())[0]
            # Get the feature dictionary for the current component
            feature_dict = component[component_type]
            
            # Check if the component type matches the provided type
            if component_type != comp_type:
                raise ValueError(f"Component type mismatch: {component_type} != {comp_type}")
            
            # Check constraints based on the component type
            if not check_component_constraints(component_type, feature_dict):
                print(component_type, feature_dict)
                return False  # Return False if any component does not meet its constraints
    
        return True  # Return True if all components meet their constraints 


    def check_component_constraints(self, component_type, comp_dict): 
        if component_type == "Granitoid":
            return self.Grainitoid_Constraint(comp_dict)
            
        elif component_type == "Shale":
            return self.Shale_Constraint(comp_dict)
            
        elif component_type == "Mafic":
            return self.Mafic_Constraint(comp_dict)
            
        elif component_type == "Halite":
            return self.Halite_Constraint(comp_dict)
            
        elif component_type == "Limestone":
            return self.Limestone_Constraint(comp_dict)
            
        elif component_type == "Anhydrite":
            return self.Anhydrite_Constraint(comp_dict)
            
        elif component_type == "Dolomite":
            return self.Dolomite_Constraint(comp_dict)
            
        elif component_type == "Random":
            return self.Random_Constraint(comp_dict)  
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    def check_all_constraints(self, components):
        """
        Check if all components meet their respective constraints.
        
        Parameters:
        components (list): A list of component dictionaries.
        
        Returns:
        bool: True if all components meet their constraints, False otherwise.
        """
        for component in components:
            component_type = list(component.keys())[0]
            feature_dict = component[component_type]
            
            if not self.check_component_constraints(component_type, feature_dict):
                return False  # Return False if any component does not meet its constraints
    
        return True  # Return True if all components meet their constraints
    

    def Prepare_Components_for_Matrix(self, components): 
        # Extract the values from each component dictionary into the array
        data_array = np.array([[comp[list(comp.keys())[0]][feature] for feature in self.Compositional_Features]
                               for comp in components])
    
        return data_array

    def Geo_Model(self, Mixture_Matrix, Components): 
        """
        Function to forward model a 
        compositional datasets for the MicroTektites

        Returns
        =======
        Array = Input Data Shape
        """
        Comps = self.Prepare_Components_for_Matrix(Components)
        Model = np.dot(Mixture_Matrix.T, Comps)
    
        return Model 

    ######################
    ## Component Priors ##
    ######################
    def Grainitoid_Priors(self, comp_dict): 
        priors = {
            'K$_{2}$O': (4.3, 2.6),
            'CaO': (1.46, 2.52),
            'TiO$_{2}$': (0.29, 0.25),
            'SiO$_{2}$': (72.16, 8.05),
            'Al$_{2}$O$_{3}$': (13.9, 3),
            'Na$_{2}$O': (3.6, 2),
            'MgO': (0.63, 0.1),
            'FeO': (1.4, 2),
            'MnO': (0.05, 0.01),
            'Cl': (0.011, 0.03),
            'SO$_{3}$': (0.05, 0.42),
            'P$_{2}$O$_{5}$': (0.11, 0.33)
        }
    
        lp = 0
        for component, (mean_val, std_val) in priors.items():
            if component in comp_dict:
                lp += norm.logpdf(comp_dict[component], mean_val, std_val)
            else:
                raise ValueError(f"Component {component} not found in comp_dict")
    
        return lp

    def Limestone_Prior(self, comp_dict):
        priors = {
            'K$_{2}$O': (1e-6, 1e-6),
            'CaO': (1, 0.03),
            'TiO$_{2}$': (1e-6, 1e-6),
            'SiO$_{2}$': (1e-6, 1e-6),
            'Al$_{2}$O$_{3}$': (1e-6, 1e-6),
            'Na$_{2}$O': (1e-6, 1e-6),
            'MgO': (1e-6, 1e-6),
            'FeO': (1e-6, 1e-6),
            'MnO': (1e-6, 1e-6),
            'Cl': (1e-6, 1e-6),
            'SO$_{3}$': (1e-6, 1e-6),
            'P$_{2}$O$_{5}$': (1e-6, 1e-6)
        }
    
        lp = 0
        for component, (mean_val, std_val) in priors.items():
            if component in comp_dict:
                lp += norm.logpdf(comp_dict[component], mean_val, std_val)
            else:
                raise ValueError(f"Component {component} not found in comp_dict")
    
        return lp

    def Shale_Prior(self, comp_dict):
        priors = {
            'K$_{2}$O': (0.04, 0.06),
            'CaO': (0.024, 0.015),
            'TiO$_{2}$': (0.0009, 0.001),
            'SiO$_{2}$': (0.6, 0.05),
            'Al$_{2}$O$_{3}$': (0.185, 0.06),
            'Na$_{2}$O': (0.018, 0.02),
            'MgO':(0.029, 0.05),
            'FeO': (0.07,0.08),
            'MnO': (0.001, 0.001),
            'Cl': (0.00001, 0.00001),
            'SO$_{3}$': (0.000501, 0.00001),
            'P$_{2}$O$_{5}$': (0.002, 0.002)
        }
    
        lp = 0
        for component, (mean_val, std_val) in priors.items():
            if component in comp_dict:
                lp += norm.logpdf(comp_dict[component], mean_val, std_val)
            else:
                raise ValueError(f"Component {component} not found in comp_dict")
    
        return lp
        
    def Mafic_Prior(self, comp_dict):
        priors = {
            'K$_{2}$O': (0.01, 0.015),
            'CaO': (0.023, 0.04),
            'TiO$_{2}$': (0.011, 0.03),
            'SiO$_{2}$': (0.435, 0.08),
            'Al$_{2}$O$_{3}$': (0.1, 0.03),
            'Na$_{2}$O': (0.006, 0.01),
            'MgO':(0.11, 0.08),
            'FeO': (0.13,0.09),
            'MnO': (0.0011, 0.002),
            'Cl': (0.00225, 0.00225),
            'SO$_{3}$': (0.0014, 0.002),
            'P$_{2}$O$_{5}$': (0.0011, 0.0015)
        }
    
        lp = 0
        for component, (mean_val, std_val) in priors.items():
            if component in comp_dict:
                lp += norm.logpdf(comp_dict[component], mean_val, std_val)
            else:
                raise ValueError(f"Component {component} not found in comp_dict")
    
        return lp 


    def Dolomite_Prior(self, scomp_dict):
        priors = {
            'K$_{2}$O': (1e-6,1e-6),
            'CaO': (0.51,0.1),
            'TiO$_{2}$': (1e-6, 1e-6),
            'SiO$_{2}$': (1e-6, 1e-6),
            'Al$_{2}$O$_{3}$': (1e-6, 1e-6),
            'Na$_{2}$O': (1e-6, 1e-6),
            'MgO':(0.4697,0.1),
            'FeO': (0.0112, 0.0212),
            'MnO': (1e-6, 1e-6),
            'Cl': (1e-6, 1e-6),
            'SO$_{3}$': (1e-6, 1e-6),
            'P$_{2}$O$_{5}$': (1e-6, 1e-6)
        }
    
        lp = 0
        for component, (mean_val, std_val) in priors.items():
            if component in comp_dict:
                lp += norm.logpdf(comp_dict[component], mean_val, std_val)
            else:
                raise ValueError(f"Component {component} not found in comp_dict")
    
        return lp

    def Halite_Prior(self, comp_dict):
        priors = {
            'K$_{2}$O': (1e-6,1e-6),
            'CaO': (1e-6,1e-6),
            'TiO$_{2}$': (1e-6, 1e-6),
            'SiO$_{2}$': (1e-6, 1e-6),
            'Al$_{2}$O$_{3}$': (1e-6, 1e-6),
            'Na$_{2}$O': (0.5, 0.1),
            'MgO':(1e-6, 1e-6),
            'FeO': (1e-6, 1e-6),
            'MnO': (1e-6, 1e-6),
            'Cl': (0.5, 0.1),
            'SO$_{3}$': (1e-6, 1e-6),
            'P$_{2}$O$_{5}$': (1e-6, 1e-6)
        }
    
        lp = 0
        for component, (mean_val, std_val) in priors.items():
            if component in comp_dict:
                lp += norm.logpdf(comp_dict[component], mean_val, std_val)
            else:
                raise ValueError(f"Component {component} not found in comp_dict")
    
        return lp

    def Anhydrite_Prior(self, comp_dict):
        priors = {
            'K$_{2}$O': (1e-6,1e-6),
            'CaO': (0.4,0.1),
            'TiO$_{2}$': (1e-6, 1e-6),
            'SiO$_{2}$': (1e-6, 1e-6),
            'Al$_{2}$O$_{3}$': (1e-6, 1e-6),
            'Na$_{2}$O':(1e-6, 1e-6),
            'MgO':(1e-6, 1e-6),
            'FeO': (1e-6, 1e-6),
            'MnO': (1e-6, 1e-6),
            'Cl': (1e-6, 1e-6),
            'SO$_{3}$': (0.6, 0.1),
            'P$_{2}$O$_{5}$': (1e-6, 1e-6)
        }
    
        lp = 0
        for component, (mean_val, std_val) in priors.items():
            if component in comp_dict:
                lp += norm.logpdf(comp_dict[component], mean_val, std_val)
            else:
                raise ValueError(f"Component {component} not found in comp_dict")
    
        return lp

    def check_component_priors(self, component_type, comp_dict): 
        if component_type == "granitoid":
            return self.Granitoid_Prior(comp_dict)
        # Add other component types here
        elif component_type == "shale":
            return self.Shale_Prior(comp_dict)
        elif component_type == "mafic":
            return self.Mafic_Prior(comp_dict)
        elif component_type == "halite":
            return self.Halite_Prior(comp_dict)
        elif component_type == "limestone":
            return self.Limestone_Prior(comp_dict)
        elif component_type == "anhydrite":
            return self.Anhydrite_Prior(comp_dict)
        elif component_type == "dolomite":
            return self.Dolomite_Prior(comp_dict)
     
        else: 
            return 0
        #else:
        #    raise ValueError(f"Unknown component type: {component_type}")
    
    def check_all_priors(self,components): 
        """
        Check if all component priors
        
        Parameters:
        components (list): A list of component dictionaries.
        
        Returns:
        bool: True if all components meet their constraints, False otherwise.
        """
        lp_values = []
        for component in components:
            component_type = list(component.keys())[0]
            feature_dict = component[component_type]
            
            lp = self.check_component_priors(component_type, feature_dict)
            lp_values.append(lp)
            
            if lp == -np.inf:
                return False, np.sum(lp_values)  # Return False if any component does not meet its constraints
    
        return True, np.sum(lp_values)  # Return True if all components meet their constraints
    """
    Prior
    """

    def logprior(self, components, mixture):
        """
        This function defines the prior 
        for the model parameters
        
        Returns
        -------
        
        LogPrior (float)
        """
        K = len(components)
        D, N = mixture.shape
        """
        Number of components prior
        """
        lp = self.Log_Prior_K(K) 
    
        
        """
        Non-negative and sum-to-one
        """
        components_val = self.Prepare_Components_for_Matrix(components)
        if np.any(components_val < 0) or np.any(mixture < 0):
            return -np.inf  # Violates non-negativity
        
        if not np.isclose(np.any(mixture.sum(axis = 1)), 1):
            return -np.inf  # Violates sum-to-one constraint
        
        """
        Prior to favour components 
        in the dictionary 
        """
        accept, comp_prior =  self.check_all_priors(components)
        if accept: 
            lp += comp_prior
        else: 
            return -np.inf
                
        return lp     

    """
    Likelihood
    """
    def loglikelihood(self, components, mixture):
        """
        This funciton takes the data and model 
        performs a centred-log transform and 
        then calculates the likelihood of the data 
        given the model for a given number of components and 
        mixture matrix
        
        Returns
        -------
        
        Log-likelihood (Float)
        """
        model = self.Geo_Model(mixture, components)
        
        self.Data_Clean[self.Data_Clean == 0] = 1e-16
        model[model == 0] = 1e-16
        
        clr_data = self.clr_transform(self.Data_Clean)
            
        clr_model = self.clr_transform(model)
        
        sigma_data = (0.5/100) * abs(clr_data)
        
        loglikelihood = - 0.5 * np.sum(((clr_model - clr_data)/sigma_data)**2)
    
        
        return loglikelihood

    """
    Posterior
    """
    def logPosterior(self, components, mixture): 
        """
        This function calculates the posterior of 
        the model.
        
        log(likelihood) + log(Prior)
        
        Returns
        -------
        LogPosterior (Float)
        """
        
        LogLike = self.loglikelihood(components, mixture)
        LogPrior = self.logprior(components, mixture)
        
        return LogLike + LogPrior


    ##############################################
    ################ MCMC MOVES ##################
    ##############################################

    def mixture_proposal(self, components, mixture, tuning_factors, index): 
        components_current = components
        mixture_current = mixture
        D, N = mixture.shape

        # Decide on the type of move
        move_type = np.random.choice(['adjust_component', 'some_components', 'whole_matrix_adjustment'])
        mixture_prime = np.copy(mixture_current)
    
        if move_type == 'adjust_component':
            if np.random.rand() < 0.5: 
                adjustment_scale = np.random.normal(0, 
                                0.05)
            else: 
                adjustment_scale = np.random.normal(0, 
                                0.1)
        
            mixture_prime[index] = np.abs(mixture_current[index] + adjustment_scale)
        
            # Normalize the mixture matrix
            mixture_prime /= mixture_prime.sum(axis=0, keepdims=True)
    
        elif move_type == 'some_components': 
            num_elements_to_adjust = np.random.randint(1, D*N)
            total_elements = D * N
    
            indices_to_adjust = np.random.choice(total_elements, num_elements_to_adjust, 
                                                 replace = False)
            adjust_value = np.random.normal(0, 0.1, num_elements_to_adjust)
    
            flat_matrix = mixture_current.flatten()
    
            flat_matrix[indices_to_adjust] += adjust_value
    
            mixture_prime = abs(flat_matrix.reshape(D,N))
            
    
        elif move_type == 'whole_matrix_adjustment':
            # Adjust the whole mixture matrix slightly
            adjustment_matrix = np.random.normal(0, 0.1, (D,N))
            mixture_prime = np.abs(mixture_current + adjustment_matrix)
            mixture_prime /= mixture_prime.sum(axis=0, keepdims=True)
    
        # Evaluate models and posteriors
        model_prime = self.Geo_Model(mixture_prime, components_current)
        model_current = self.Geo_Model(mixture_current, components_current)
    
        current_posterior = self.logPosterior(components_current, mixture_current)
        proposed_posterior = self.logPosterior(components_current, mixture_prime)
    
        u = np.random.rand()
        if proposed_posterior > current_posterior or np.log(u) < (proposed_posterior - current_posterior):
            return mixture_prime, components_current, proposed_posterior, model_prime, True
        else:
            return mixture_current, components_current, current_posterior, model_current, False


    def component_proposal(self, components, mixture):
        """
        This function changes the composition of an  
        end-member
        
        Returns
        -------
        Mixture, Components, Posterior, Model
        """
        components_current = components
        mixture_current = mixture
        current_posterior = self.logPosterior(components_current, 
                                         mixture_current)
        
        # Choose a random component to modify
        i = random.randint(0, len(components) - 1)
        selected_component = components[i]
        component_type = list(selected_component.keys())[0]
        feature_dict = selected_component[component_type]
    
        # Select a random feature to change
        feature_keys = list(feature_dict.keys())
        j = random.choice(feature_keys)
    
        # Make a copy of the components to propose changes
        components_prime = [dict(comp) for comp in components]  # Deep copy each component
        feature_prime = dict(feature_dict)  # Copy the selected component's features
        components_prime[i] = {component_type: feature_prime}
    
    
        # Modify the selected feature
        feature_prime[j] += np.random.normal(0, 0.01 * feature_prime[j])
        feature_prime[j] = abs(feature_prime[j])  # Ensure non-negativity
    
        # Normalize the modified component's features to sum to 1
        total = sum(feature_prime.values())
        for key in feature_keys:
            feature_prime[key] /= total
    
    
        if not self.check_all_constraints(components_prime):
            # If the constraints are not met, return the current state
            model_current = self.Geo_Model(mixture_current, components_current)
            return mixture_current, components_current, current_posterior, model_current, False
    
        
        proposed_posterior = self.logPosterior(components_prime, 
                                         mixture)
    
        model_prime = self.Geo_Model(mixture, 
                                components_prime)
        
        model_current = self.Geo_Model(mixture, 
                                  components_current)
        
        u = np.random.rand()
        if proposed_posterior > current_posterior or np.log(u) < (proposed_posterior - current_posterior):
            mixture = mixture
            components = components_prime
            posterior = proposed_posterior
            model = model_prime
            return mixture, components, posterior, model, True
    
            
        else:
            mixture = mixture_current
            components = components_current
            posterior = current_posterior
            model = model_current
            
            return mixture, components, posterior, model, False

    def component_swap(self, components, mixture):
        """
        This function suggests swapping the composition of one or more 
        end-members with no change in dimensionality.

        Returns
        -------
        Mixture, Components, Posterior, Model
        """
        
        components_current = components
        mixture_current = mixture
        K = len(components)
        current_posterior = self.logPosterior(components_current, mixture_current)
        
        # Determine the number of components to modify (1 to max_k)
        num_swaps = random.randint(1, min(K, K))
        
        # Select random components to modify
        indices_to_modify = random.sample(range(K), num_swaps)
        
        components_prime = [dict(comp) for comp in components]  # Deep copy each component
        
        for i in indices_to_modify:
            # Choose a component from the component dictionary
            random_component_key = random.choice(list(self.component_dictionary.keys()))
            components_prime[i] = {random_component_key: self.component_dictionary[random_component_key]}  # Swap with a component from the dictionary
        
        model_prime = self.Geo_Model(mixture, components_prime)
        model_current = self.Geo_Model(mixture, components_current)
    
        if not self.check_all_constraints(components_prime):
            # If the constraints are not met, return the current state
            return mixture_current, components_current, current_posterior, model_current, False
        
        """
        Acceptance
        """
        
        proposed_posterior = self.logPosterior(components_prime, mixture)
        
        u = np.random.rand()
        if proposed_posterior > current_posterior or np.log(u) < (proposed_posterior - current_posterior):
            return mixture, components_prime, proposed_posterior, model_prime, True
        else:
            return mixture_current, components_current, current_posterior, model_current, False    

    def mixture_swap(self, components, mixture):
        """
        This function suggest swapping a component 
        with another, keeping all other variables the same
        
        Returns
        -------
        Mixture, Components, Posterior, Model
        """
        components_current = components
        mixture_current = mixture
        
        mixture_prime = np.copy(mixture)
        
    
        i, j = np.random.choice(len(mixture_prime), 
                                size=2, 
                                replace=False)
        
        selection = np.random.uniform()
        
        if selection < 0.5: 
            mixture_prime[i], mixture_prime[j] = mixture_prime[j], mixture_prime[i]
        else: 
            mixture_prime[i, :], mixture_prime[j, :] = mixture_prime[j, :], mixture_prime[i, :]
        
        
        model_prime = self.Geo_Model(mixture_prime, components_current)
        
        model_current = self.Geo_Model(mixture, components_current)
        
        """
        Acceptance
        """
        current_posterior = self.logPosterior(components_current, 
                                         mixture_current)
        
        proposed_posterior = self.logPosterior(components_current, 
                                         mixture_prime)
     
        u = np.random.rand()
        if proposed_posterior > current_posterior or np.log(u) < (proposed_posterior - current_posterior):    
            mixture = mixture_prime
            components = components_current
            posterior = proposed_posterior
            model = model_prime
            return mixture, components, posterior, model, True
        else:
            mixture = mixture_current
            components = components_current
            posterior = current_posterior
            model = model_current
            
        return mixture, components, posterior, model, False

    def select_random_component(self):
        """
        Randomly selects one component from the component dictionary
        """
        key = random.choice(list(self.component_dictionary.keys()))
        return key, self.component_dictionary[key]


    def component_birth(self, components, mixture): 
        """
        This function defines the birth proposal function
        
        Returns
        -------
        Mixture
        Components
        Posterior
        Model
        """
        components_current = components
        mixture_current = mixture
        current_posterior = self.logPosterior(components_current, 
                                         mixture_current)
        N = mixture.shape[1]
        
        """
        New component 
         - Select from dictionary 
        """
        j = np.random.choice(self.select_random_component())
        selected_component_key = random.choice(list(self.component_dictionary.keys()))
        component_new = {selected_component_key: self.component_dictionary[selected_component_key]}
    
        """
        New Mixing vector
        """
        alpha2 = np.ones(N)
        mixing_new = np.random.dirichlet(alpha2, 1).reshape(1, N)
        
        """
        Add this into the component matrix
        and mixing matrix
        """
        
        components_prime = components_current + [component_new]   
    
        
        mixture_prime = np.concatenate([mixture,
                                        mixing_new],
                                         axis = 0)    
    
        if not self.check_all_constraints(components_prime):
            # If the constraints are not met, return the current state
            model_current = self.Geo_Model(mixture_current, components_current)
            return mixture_current, components_current, current_posterior, model_current, False
        
        model_prime = self.Geo_Model(mixture_prime, components_prime)
        
        model_current = self.Geo_Model(mixture, components_current)
        
        """
        Acceptance
        """
        current_posterior = self.logPosterior(components_current, 
                                         mixture_current)
        
        proposed_posterior = self.logPosterior(components_prime, 
                                         mixture_prime)
        
        u = np.random.rand()
        if proposed_posterior > current_posterior or np.log(u) < (proposed_posterior - current_posterior):
            mixture = mixture_prime
            components = components_prime
            posterior = proposed_posterior
            model = model_prime
            return mixture, components, posterior, model, True
            
        else:
            mixture = mixture_current
            components = components_current
            posterior = current_posterior
            model = model_current
            
            return mixture, components, posterior, model, False



    def component_death(self,components, mixture): 
        """
        This function defines the death proposal function
        
        Returns
        -------
        Mixture, Components, Posterior, Model
        """
        components_current = components
        mixture_current = mixture
        K = len(components)
        
        # Select a component to delete
        j = random.randint(0, K - 1)
        
        # Add this into the component matrix and mixing matrix
        components_prime = [comp for idx, comp in enumerate(components) if idx != j]
        mixture_prime = np.delete(mixture, j, axis=0)
        mixture_prime /= mixture_prime.sum(axis=0)
    
    
        model_prime = self.Geo_Model(mixture_prime, components_prime)
        model_current = self.Geo_Model(mixture, components_current)
        
        
        """
        Acceptance
        """
        current_posterior = self.logPosterior(components_current, 
                                         mixture_current)
        
        proposed_posterior = self.logPosterior(components_prime,
                                         mixture_prime)
    
    
        if not self.check_all_constraints(components_prime):
            # If the constraints are not met, return the current state
            model_current = self.Geo_Model(mixture_current, components_current)
            return mixture_current, components_current, current_posterior, model_current, j, False
        
        
        u = np.random.rand()
        if proposed_posterior > current_posterior or np.log(u) < (proposed_posterior - current_posterior):
            mixture = mixture_prime
            components = components_prime
            posterior = proposed_posterior
            model = model_prime
            return mixture, components, posterior, model, j, True
            
        else:
            mixture = mixture_current
            components = components_current
            posterior = current_posterior
            model = model_current
            
            return mixture, components, posterior, model, j, False

    def component_merge(self, components, mixture): 
        """
        This function defines a merge proposal where the model selects
        two similar components and merges their compositions and mixture 
        weights. Dimensionality reduces by 1. 
        
        Returns
        -------
        Mixture, Components, Posterior, Model
        """
        components_current = components
        mixture_current = mixture
        current_posterior = self.logPosterior(components_current, 
                                         mixture_current)
    
        K = len(components)
        
        # Find components of the same type
        component_types = [list(comp.keys())[0] for comp in components]
        unique_types = list(set(component_types))
        
        # Identify pairs of components of the same type
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K) if component_types[i] == component_types[j]]
    
        if not pairs:
            return mixture, components, self.logPosterior(components, mixture), self.Geo_Model(mixture, components), None, False
        # Randomly select a pair to merge
        i, j = random.choice(pairs)
        
        # Merge Components
        component_type = component_types[i]
        component_i = components[i][component_type]
        component_j = components[j][component_type]
        
        merged_component = {k: (component_i[k] + component_j[k]) / 2 for k in component_i.keys()}
        component_merge = {component_type: merged_component}
        
        # Merge Mixture
        mixture_merge = (mixture[i] + mixture[j]) / 2
        mixture_merge = mixture_merge.reshape(1, -1)
    
        # Delete the components
        mixture_delete = np.delete(mixture_current, [i, j], axis=0)
        components_delete = [comp for idx, comp in enumerate(components_current) if idx not in [i, j]]
        
        # New Components and Mixture
        components_prime = components_delete + [component_merge]
        mixture_prime = np.concatenate([mixture_delete, mixture_merge], axis=0)
    
        # Normalize the mixture_prime
        mixture_prime /= mixture_prime.sum(axis=0)
    
        if not self.check_all_constraints(components_prime):
            # If the constraints are not met, return the current state
            model_current = self.Geo_Model(mixture_current, components_current)
            return mixture_current, components_current, current_posterior, model_current, None, False
    
        
        
        model_prime = self.Geo_Model(mixture_prime, components_prime)
        model_current = self.Geo_Model(mixture, components_current)
        
        # Acceptance
        proposed_posterior = self.logPosterior(components_prime, mixture_prime)
        
        u = np.random.rand()
        if proposed_posterior > current_posterior or np.log(u) < (proposed_posterior - current_posterior):
            mixture = mixture_prime
            components = components_prime
            posterior = proposed_posterior
            model = model_prime
            return mixture, components, posterior, model, (i, j), True
        else:
            mixture = mixture_current
            components = components_current
            posterior = current_posterior
            model = model_current
            return mixture, components, posterior, model, (i, j), False

    def initialize_Mixture_Model(self, component_dict, min_comp, max_comp):
        N = np.random.randint(min_comp, max_comp)
        # Randomly select components from the component dictionary
        selected_components = self.randomly_select_components(n_samples=N)
        
        Mixture_Matrix_guess = np.random.dirichlet(np.ones(self.Data_Clean.shape[0]), N)
        
        return selected_components, Mixture_Matrix_guess


    def Initial_Guesses_for_Model(self):
        In_models = []
        for i in range(self.n_chains): 
            log_post = -np.inf
            Checking = False
            while True: 
                Initial_comp, Initial_mix = self.initialize_Mixture_Model(self.component_dictionary,
                                                                          self.minimum_components, 
                                                                          self.maximum_components)
                Checking = self.check_all_constraints(Initial_comp)
                log_post = self.logPosterior(Initial_comp, Initial_mix)
        
                if log_post != -np.inf and Checking == True:
                    In_models.append((Initial_comp, Initial_mix))
                    break  # Exit the loop since both conditions are satisfied
                    
        return In_models

    def Print_Initial_Posteriors(self): 
        Initial_Guesses = self.Initial_Guesses_for_Model()
        for i in range(self.n_chains): 
            print(f'Initial_{i}', self.logPosterior(Initial_Guesses[i][0],
                                             Initial_Guesses[i][1]), 
                  self.check_all_constraints(Initial_Guesses[i][0]))

    """
    Other Functions for MCMC
    """
    def adjust_tuning_factors(self,tuning_factors, 
                              current_acceptance_rate,
                              target=0.234, 
                              increase_factor=1.25, 
                               decrease_factor=0.75):
    
        adjustment_factor = np.where(current_acceptance_rate > target, increase_factor, decrease_factor)
        new_tuning_factors = tuning_factors * adjustment_factor
        # Ensure tuning factors remain within sensible bounds
        new_tuning_factors = np.clip(new_tuning_factors, a_min=0.0000001, a_max=1000000.0)
        return new_tuning_factors


    def expand_tuning_factors(self, 
                             tuning_factors, 
                             initial_tuning_value=0.1):
        # Assuming tuning_factors is a 2D array with shape (K, N)
        K, N = tuning_factors.shape
        new_row = np.full((1, N), initial_tuning_value)  # Create a row for the new component
        tuning_factors = np.vstack((tuning_factors, new_row))  # Add the new row
        return tuning_factors  


    def contract_tuning_factors(self, tuning_factors, i):
        # Remove the i-th row corresponding to the removed component
        tuning_factors = np.delete(tuning_factors, i, axis=0)
        return tuning_factors

    def adjust_tuning_factors_for_merge(self, tuning_factors, 
                                        indices, initial_tuning_value=0.1):
        """
        Adjust the tuning factors array after merging two components.
        """
        i, j = indices
    
        # Remove the tuning parameters for the merged components
        tuning_factors = np.delete(tuning_factors, [i, j], axis=0)
    
        # Add a new tuning parameter for the merged component
        new_row = np.full((1, tuning_factors.shape[1]), initial_tuning_value)
        tuning_factors = np.vstack([tuning_factors, new_row])
    
        return tuning_factors


# In[4]:


get_ipython().system("jupyter nbconvert --to script 'Bayesian_MT_Methods.ipynb'")


# In[ ]:





# In[ ]:





# In[ ]:




