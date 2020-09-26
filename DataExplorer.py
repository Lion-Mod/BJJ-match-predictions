import pandas as pd
import numpy as np

class DataExplorer:
    def __init__(self, data, cat_features, cont_features):
        """
          Used to for exploring data and suggestions to transform the data
                
          Attributes:
          - data = pandas dataframe
          - cat_features = list of categorical features in data
          - cont_features = list of continuous features in data
        """
        self.data = data
        self.features = data.columns
        self.cat_features = cat_features
        self.cont_features = cont_features

  ### 1. START INSPECTOR MISSING - FIND MISSING VALUES ###
    def missing_values_perc(self, features_to_check = None):
      """ 
        Return one table (sorted descending, 0% excluded)
        1. features_to_check = features with missing values and how much (as %) is missing

        Params:
        - features_to_check (list) = defaults to all features, if not then list of column names to check   
      """
      # Default to check all features
      if features_to_check == None:
        features_to_check = self.features
        features_to_drop = []

      # If not default then use features from the input
      else:
        features_to_drop = list(set(self.features) - set(list(features_to_check)))

      temp = self.data.drop(features_to_drop, axis = 1)

      # Create a table sorting highest to lowest missing value percentages in data
      data_missing_percs = pd.concat([(temp.isnull().sum() /  temp.isnull().count())*100], 
                                    axis = 1, 
                                    keys = ['percentage_missing']) \
                                    .sort_values('percentage_missing', ascending = False)

      # Return these percentages greater than 0 (features with missing values)
      return(data_missing_percs[data_missing_percs['percentage_missing'] > 0])
      

    def get_excessive_missing_features(self, threshold = None):
      """ 
        Get list of features with {threshold}%+ missing values, defaults to no threshold
      
        Params:
        - threshold (default = 0+) = value between 0-100, filters to features with missing percentage above threshold
      """
      # Default to lowest threshold (greater than 0 missing)
      if threshold == None:
        threshold = 0

      # If not default then use threshold
      else:
        pass

      # Get % missing values in each feature in "full" dataset
      feature_missing_percentages = pd.concat([(self.data.isnull().sum() /  self.data.isnull().count())*100], 
                                              axis = 1, keys=['percentage_missing'])

      # Return list of variables with {threshold}%+ missing values (threshold not included)
      return list(feature_missing_percentages[feature_missing_percentages['percentage_missing'] > threshold].index)


    def drop_excessive_missing_features (self, threshold = None):
      """ 
        Drop features with {threshold}%+ missing values in data
      
        Params:
        - threshold = value between 0-100, filters to features with missing percentage above threshold
      """
      
      # Get list of features that have threshold%+ missing values in data
      features_to_drop = self.get_excessive_missing_features(threshold)

      # Loop through data's features dropping features with threshold%+ missing values
      for feature in features_to_drop:
        self.data.drop(feature, axis = 1, inplace = True)

      # Output which features were dropped
      print(f"Function has removed the following", features_to_drop)
          
    def fill_missing_cat_features (self):
      """ 
        Fill categorical features in the data with "NONE" then convert to string
      """

      # Loops through each categorical feature in data and fills NAs with "NONE" then converts to string
      for feature in self.cat_features:
         self.data.loc[:, feature] = self.data[feature].fillna("NONE").astype(str)

      print("nans replaced with 'NONE'")

    def fill_missing_cont_features (self, fill_missing):
      """
        Fill continuous features in the data with a user defined value
      """

      # Loops through each continuous feature column in the data and fills NAs with fill_missing
      for feature in self.cont_features:
        self.data.loc[:, feature] = self.data[feature].fillna(fill_missing)

      print("nans replaced with", fill_missing)

  ### END INSPECTOR MISSING - FIND MISSING VALUES ###

  ### 2. START BINNER - find potentially optimum bins ###   
    def freedman_diaconis(self):
        """
          Print each continuous column's bin width based upon freedman diaconis formula
        """
        print("=== CONTINUOUS FEATURES FREEDMAN BINNING WIDTHS ===")

        # Loop through continuous features and calculate bin widths using freedman_diaconis formula
        # - Find the lower quartile for the feature
        # - Find the upper quartile for the feature
        # - Find the IQR
        # - Get the size of the data
        for cont_feature in self.cont_features:
            Q1 = self.data[cont_feature].quantile(0.25)
            Q3 = self.data[cont_feature].quantile(0.75)
            IQR = Q3 - Q1
            N   = self.data.size
            bin_width  = (2 * IQR) / np.power(N, 1/3)

            # Output feature and the bin_width
            print(cont_feature, ":", bin_width)
        
        print("\n")

    ### END BINNER ###        

    ### 3. START TREASURE HUNTER - FINDING POTENTIALLY RARE VALUES ###
    def get_rare_values(self, features_to_rarify, threshold = 1, mandatory_levels = 1):
        """
        Pull features and their levels that need to be rarified (based upong threshold) + highlight any features that don't meet mandatory_levels
        
        Params:
        - threshold = return the level if level makes up %threshold or less in the features
        - features_to_rarify = categorical features to check
        - mandatory_levels = int where only features with at least mandatory levels number of levels will be returned (would you rare feature with 2 levels?)
        """

        # Initialise list for storing features with not enough levels
        not_enough_levels = []

        print("=== FEATURES AND THEIR POTENTIALLY RARE FEATURES ===")

        # Loop through each categorical feature and print levels within it that make up threshold or below % of the values
        for feature in features_to_rarify:

            # Get number of levels for feature
            num_of_levels = len(self.data[feature].drop_duplicates())

            # If there are more than or equal mandatory_levels, if so proceed
            if num_of_levels >= mandatory_levels:

                # Print the number of levels the feature has
                print(feature, "has", num_of_levels, "levels.")
                
                # Create a table of the % each level makes up of the feature
                # - Select the feature we want
                # - Fill nas
                # - Convert to string
                # - Get percentages each level makes up of feature
                # - Renaming columns
                temp = self.data[feature] \
                                        .fillna("NONE") \
                                        .astype(str) \
                                        .value_counts(normalize = True) \
                                        .rename_axis("level") \
                                        .reset_index(name = "percentage_in_data")

                # Print only those levels that make up threshold % or less (i.e. "rare" values)
                print(temp[temp["percentage_in_data"] <= (threshold)], "\n")
            
            # Else add the feature to a list of features not above mandatory_levels
            else:
                not_enough_levels.append(feature)

        # Print list of features not having the mandatory_levels
        print("The following features didn't meet the mandatory levels:", not_enough_levels)
    ### END TREASURE HUNTER ###
