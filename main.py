## import libraries -----------------------------
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import sys
from operator import itemgetter


class MovieRecommendation():

    def __init__(self):
        # minimum support for movies 
        self.min_support = 50
        # Definning a minimum for confidence level
        self.min_confidence = 0.9
        self.frequent_itemsets = {}
        # Dataset placeholders
        self.ratings_full = {}
        self.ratings = {} # Minimise version of the whole dataset
        self.movies  = {}
        self.users_movies = {}
        self.significant_rules = {}


    def load_data(self):
        ###############
        ## Setting path
        ###############  
        data_folder = os.path.join(os.path.curdir,"input")

        rating_filename = os.path.join(data_folder, "ratings.dat")
        movie_name_filename = os.path.join(data_folder, "movies.dat")

        # Defining dateparser for reading date column
        dateparse = lambda x: pd.to_datetime(x, unit='s')

        # Reading the reviews file and defining the columns name
        self.ratings_full = pd.read_csv(rating_filename, 
                                        delimiter="::", 
                                        header=None, 
                                        names=['UserID','MovieID','Rating','Datetime'], 
                                        parse_dates=['Datetime'],
                                        date_parser=dateparse,
                                        engine='python')

        # Reading movies dat file and setting column names
        self.movies = pd.read_csv(movie_name_filename, 
                                  delimiter="|", 
                                  header=None, 
                                  encoding = "mac-roman")

        self.movies.columns = ['MovieID','Title','Genres']


    def data_prepration(self, rating_threshold = 3, user_threshold = 200):
        # Adding Favorable feature if user rated over 3
        self.ratings_full['Favorable'] = self.ratings_full['Rating'] > rating_threshold

        # Make a sample dataset to make our Apriori algorithm faster
        self.ratings = self.ratings_full[self.ratings_full.UserID <= user_threshold]

        # Filtering the dataset for only favorable movies
        favorable_ratings = self.ratings[self.ratings['Favorable']]

        # List of movies which each user considered as favorable
        self.users_movies = dict((user_id, frozenset(movies)) #why frozenset, only cuz of speed in search 
                                 for user_id, movies in \
                                 favorable_ratings.groupby("UserID")["MovieID"])


    def create_initial_freq_itemsets(self):
        # Frequency of each movie given a favorable review
        movie_freq = self.ratings[['MovieID', 'Favorable']].groupby('MovieID').sum()

        # superset_length=1 candidates are the isbns with more than min_support favourable reviews
        self.frequent_itemsets[1] = dict((frozenset((movie_id,)), row["Favorable"]) \
                                         for movie_id, row in movie_freq.iterrows() \
                                         if row["Favorable"] > self.min_support)

        print("{} frequent itemsets in length of {}".format(len(self.frequent_itemsets[1]), 1))
        sys.stdout.flush()


    def create_supersets(self, superset_max_size = 15):        
        for superset_length in range(2, superset_max_size):
            # Generate candidates of length superset_length, using the frequent itemsets of its precedure
            candidate_superset_freq = defaultdict(int)

            for user_id, user_movies in self.users_movies.items():
                # Check if each itemset in the last freq_itemsets exist in user movies 
                for itemset in self.frequent_itemsets[superset_length-1]:
                    if itemset.issubset(user_movies):
                        # Construst next level superset based other favoriate movies of users exclude the current itemset itself 
                        for other_reviewed_movie in user_movies - itemset:
                            current_superset = itemset | frozenset((other_reviewed_movie,))
                            # increase the freqency of recent superset which just occured
                            candidate_superset_freq[current_superset] += 1

            new_frequent_itemsets = dict([(itemset, frequency) \
                                         for itemset, frequency in candidate_superset_freq.items() if frequency >= self.min_support])

            
            if len(new_frequent_itemsets):
                print("{} frequent itemsets in length of {}".format(len(new_frequent_itemsets), superset_length))
                sys.stdout.flush()
                self.frequent_itemsets[superset_length] = new_frequent_itemsets
            
            elif len(new_frequent_itemsets) == 0:
                print("There is no frequent itemsets in length of {}".format(superset_length))
                sys.stdout.flush()
                break

        # Itemsets in length 1 are not useful for recommending system so we can drop it
        del self.frequent_itemsets[1]
        
        print("Total numer of itemsets in any length: {0}".format(sum(len(itemsets) for itemsets in self.frequent_itemsets.values())))



    def extract_association_rules(self):
        '''
        Our approach for identifying rules are simle,
        iterating over freq_itemsets and within each itemset
        excluding each member of set and considering it as conclusion
        and all remaining as premises.
        '''
        candidate_rules = []
        for itemset_length, itemset_counts in self.frequent_itemsets.items():
            for itemset in itemset_counts.keys():
                for conclusion in itemset:
                    premise = itemset - set((conclusion,))
                    candidate_rules.append((premise, conclusion))

        # Next, we compute the confidence of each of these rules.
        valid_rule   = defaultdict(int)
        invalid_rule = defaultdict(int)

        for user, fav_movies in self.users_movies.items():
            for candidate_rule in candidate_rules:
                premise, conclusion = candidate_rule
                # If user liked all premise movies
                if premise.issubset(fav_movies):
                    # If user liked conclusion movie too
                    if conclusion in fav_movies:
                        # Then rule should be considered as valid
                        valid_rule[candidate_rule] += 1
                    else:
                        invalid_rule[candidate_rule] += 1

        # Calculating confidence rule for all the extracted rules                
        rule_confidence = {candidate_rule: valid_rule[candidate_rule] / float(valid_rule[candidate_rule] + invalid_rule[candidate_rule])
                           for candidate_rule in candidate_rules}

        # Filter out the rules with poor confidence
        self.significant_rules = {rule: confidence for rule, confidence in rule_confidence.items() if confidence > self.min_confidence}
        
        print("In total {} rules found which {} of them are significant.".format(len(candidate_rules), len(self.significant_rules)))


    def report_associations(self):
        sorted_confidence = sorted(self.significant_rules.items(), key=itemgetter(1), reverse=True)
        for index in range(5):
            print("Rule #{0}".format(index + 1))
            (premise, conclusion) = sorted_confidence[index][0]
            print("Rule: If a person recommends {0} they will also recommend {1}".format(premise, conclusion))
            print(" - Confidence: {0:.3f}".format(self.significant_rules[(premise, conclusion)]))
            print("")
