import pandas as pd
import numpy as np

from sklearn import preprocessing
from scipy import stats

import numpy as np
import pandas as pd

class MissingDataObserver:
    def __init__(self, train_df, valid_df, target):
        """
        Used to find missing values in both training and validation set
                
        Inputs
            - train_df = pandas dataframe
            - valid_df = pandas dataframe
            - target = target column
        """
        self.train_df = train_df
        self.valid_df = valid_df
        self.target = target

    def drop_target_column(self):
      """ 
        Used in missing_values_perc() and get_excessive_missing_features()
      """
      # Create a copy of training dataframe
      train_copy = self.train_df.copy()

      train_no_target = train_copy.drop(self.target, axis = 1)
      # Drop target column and return dataframe
      return train_no_target


    def missing_values_perc(self):
      """ 
        Return two tables (sorted descending, 0% excluded)
            1. train_df's features with missing values and how much (as %) is missing
            2. valid_df's features with missing values and how much (as %) is missing
      """
      # Drop target column from training dataset
      temp = self.drop_target_column()

      # Create a table sorting highest to lowerest missing value percentages in train_df
      train_missing_perc = pd.concat([(temp.isnull().sum() /  temp.isnull().count())*100], 
                                    axis = 1, 
                                    keys = ['percentage_missing']) \
                                    .sort_values('percentage_missing', ascending = False)

      # Create a table sorting highest to lowerest missing value percentages in valid_df
      valid_missing_perc = pd.concat([(self.valid_df.isnull().sum() /  self.valid_df.isnull().count())*100], 
                                      axis = 1, 
                                      keys = ['percentage_missing']) \
                                      .sort_values('percentage_missing', ascending = False)

      # Return these percentages greater than 0 (columns with missing values)
      return(train_missing_perc[train_missing_perc['percentage_missing'] > 0],
            valid_missing_perc[valid_missing_perc['percentage_missing'] > 0])
      

    def get_features_with_missing_value(self):
      """ 
        Return two lists
          1. List of train_df features with missing values
          2. List of valid_df features with missing values
      """
      train_missing, valid_missing = self.missing_values_perc()

      # Return two lists
      # 1. train_df's columns with missing values
      # 2. valid_df's columns with missing values
      return(list(train_missing.index), 
             list(valid_missing.index))


    def get_excessive_missing_features(self, threshold):
      """ 
      Get list of features with {threshold}%+ missing values in "full" dataset (train_df + valid_df)
      
        Parameters used
          - threshold = value between 0-100, filters to features with missing percentage above threshold
      """

      # Drop target column from training dataset
      temp = self.drop_target_column()

      # Union training and valid dataset
      full = pd.concat([temp, self.valid_df])

      # Get % missing values in each column in "full" dataset
      column_missing_percentages = pd.concat([(full.isnull().sum() /  full.isnull().count())*100], 
                                              axis = 1, keys=['percentage_missing'])

      # Return list of variables with {threshold}%+ missing values (threshold not included)
      return list(column_missing_percentages[column_missing_percentages['percentage_missing'] > threshold].index)


    def drop_excessive_missing_features (self, threshold):
      """ 
        Drop columns with {threshold}%+ missing values in "full" dataset (training + valid)
      
        Parameters used
            - threshold = value between 0-100, filters to features with missing percentage above threshold
      """
      
      # Get list of features that have THRESHOLD missing values in full dataset
      features_to_drop = self.get_excessive_missing_features(threshold)

      # Loop through training and valid dataset, dropping features with THRESHOLD missing values
      for feature in features_to_drop:
        self.train_df.drop(feature, axis = 1, inplace = True)
        self.valid_df.drop(feature, axis = 1, inplace = True)

      # Output which features were dropped
      print(f"Function has removed the following", features_to_drop)
          
    def fill_missing_cat_features (self, cat_features):
      """ 
        Fill categorical variables in training and valid dataset with "NONE" then convert to string

        Parameters used
            - cat_features = list of categorical features to fill missing
      """

      # Loops through each categorical feature column in train_df/valid_df and fills NAs with "NONE" then converts to string
      for feature in cat_features:
         self.train_df.loc[:, feature] = self.train_df[feature].fillna("NONE").astype(str)
         self.valid_df.loc[:, feature] = self.valid_df[feature].fillna("NONE").astype(str)

class Binner:
    def __init__(self, data, cont_features = None):
        """
        Used to find optimal bin widths

        Inputs
            - data = pandas dataframe
            - cont_features = list of continuous features in the dataframe
        """
        self.cont_data = data[cont_features]
        self.cont_features = cont_features

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
            Q1 = self.cont_data[cont_feature].quantile(0.25)
            Q3 = self.cont_data[cont_feature].quantile(0.75)
            IQR = Q3 - Q1
            N   = self.cont_data.size
            bin_width  = (2 * IQR) / np.power(N, 1/3)

            # Output feature and the bin_width
            print(cont_feature, ":", bin_width)
        
        print("\n")    


class TreasureHunter:
    def __init__(self, data, cat_features = None):
        """
        Used to find rare levels within features

        Inputs
            - data = pandas dataframe
            - cat_features = list of categorical features in the dataframe
        """
        self.cat_data = data[cat_features]
        self.cat_features = cat_features

    def get_rare_values(self, features_to_rarify, threshold = 1, mandatory_levels = 1):
        """
        Pull features and their levels that need to be rarified + highlight any features that don't meet mandatory_levels
        
        Inputs
            - threshold = return the level if makes up %threshold or less in the features
            - features_to_rarify = categorical features to check
            - mandatory_levels = int where only features with at least mandatory levels number of levels will be returned (would you rare feature with 2 levels?)
        """
        # Initialise list for storing features with not enough levels
        not_enough_levels = []

        print("=== FEATURES AND THEIR POTENTIALLY RARE FEATURES ===")

        # Loop through each categorical feature and print levels within it that make up threshold or below % of the values
        for feature in features_to_rarify:

            # Get number of levels for feature
            num_of_levels = len(self.cat_data[feature].drop_duplicates())

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
                temp = self.cat_data[feature] \
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