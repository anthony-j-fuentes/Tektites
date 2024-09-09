#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


class Bayes_MT_input_data: 
    def __init__(self, filepath): 
        self.filepath = filepath
        self.Input_Data = self.load_data()
        self.Data_Features = ['K2O', 'CaO', 'TiO2', 
                              'SiO2','Al2O3', 'Na2O',
                              'MgO', 'FeO', 'MnO', 
                              'Cl', 'SO3', 'P2O5']

    def load_data(self): 
        """
        Load the data from the selected filepath
        """
        try: 
            self.Input_Data = pd.read_excel(self.filepath, skiprows = 3)
            return self.Input_Data
            print('MicroTektite Data is Loaded')
        except Exception as e: 
            print("Error Loading Data, Check file path and try again")


    def Unpack_Data_and_Features(self): 
        # Function to unpack data and 
        # clean it up to be used in a 
        # Compositional model 
        if self.Input_Data is None:
            print("No data to process")
            return None
        MT_data = self.Input_Data
        g_data =  MT_data[self.Data_Features].values
        g_data_clean = np.nan_to_num(g_data)
        Clean_Data = np.array(g_data_clean, copy = True)
        # Replace all non-positive values with 1e-11
        Clean_Data[Clean_Data <= 0] = 1e-11
        # Re-normalize everyting for consistency 
        Clean_Data /= Clean_Data.sum(axis = 1, 
                         keepdims = True)

        self.MT_Data_clean_up = Clean_Data

        return self.MT_Data_clean_up


# In[3]:


get_ipython().system("jupyter nbconvert --to script 'Bayesian_MT_Analysis_Input_Data.ipynb'")


# In[20]:




