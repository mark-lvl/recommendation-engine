# import libraries
import pandas as pd
import os
import sys
from operator import itemgetter
from collections import defaultdict


class MovieRecommendation:
    user_threshold: int
    min_support: int
    min_confidence: float

    def __init__(self):
        # users in train set
        self.user_threshold = 200
        # minimum support for movies 
        self.min_support = 50
        # Defining a minimum for confidence level
        self.min_confidence = 0.9
        self.itemsets = {}
        # Dataset placeholders
        self.ratings_full = {}
        self.ratings = {}  # Minimise version of the whole dataset
        self.movies = {}
        self.users_movies = {}
        self.significant_rules = {}

    def load_data(self):
        # Setting path
        data_folder = os.path.join(os.path.curdir, "input")
        rating_filename = os.path.join(data_folder, "ratings.dat")
        movie_name_filename = os.path.join(data_folder, "movies.dat")

        # Defining dateparser for reading date column
        date_parser = lambda x: pd.to_datetime(x, unit='s')

        # Reading the reviews file and defining the columns name
        self.ratings_full = pd.read_csv(rating_filename,
                                        delimiter="::",
                                        header=None,
                                        names=['UserID', 'MovieID', 'Rating', 'Datetime'],
                                        parse_dates=['Datetime'],
                                        date_parser=date_parser,
                                        engine='python')

        # Reading movies dat file and setting column names
        self.movies = pd.read_csv(movie_name_filename,
                                  delimiter="::",
                                  header=None,
                                  encoding="mac-roman",
                                  engine='python')

        self.movies.columns = ['MovieID', 'Title', 'Genres']

    def data_prep(self, rating_threshold=3, user_threshold=200):
        # Adding Favorable feature if user rated over 3
        self.ratings_full['Favorable'] = self.ratings_full['Rating'] > rating_threshold

        # Make a sample dataset to make our Apriori algorithm faster
        self.ratings = self.ratings_full[self.ratings_full.UserID <= user_threshold]

        # Filtering the dataset for only favorable movies
        favorable_ratings = self.ratings[self.ratings['Favorable']]

        # List of movies which each user considered as favorable
        self.users_movies = dict((user_id, frozenset(movies))  # why frozenset, only cuz of speed in search
                                 for user_id, movies in \
                                 favorable_ratings.groupby("UserID")["MovieID"])

    def create_initial_itemset(self):
        # Frequency of each movie given a favorable review
        movie_freq = self.ratings[['MovieID', 'Favorable']].groupby('MovieID').sum()

        """
        The structure of itemsets will be as a dictionary with following format:
        
        Structure:
            {length_of_itemset: {(set_of_movies_list_in_current_itemset): frequency_of_itemset},}
        Key: int
            length of itemset
        Value: dict
            Key: frozenset  
                a frozenset of list of involving movies in current itemset
            Value: int
                how many times current combination of movies occurred in user ratings
            
        Example:
            for a itemset comprises of 3 movies
            {(movie_1,movie_15,movie_495) : 59}
        """

        # itemset_length=1 are a list of all movies which have rating more than min_support
        self.itemsets[1] = dict((frozenset((movie_id,)), row["Favorable"])
                                for movie_id, row in movie_freq.iterrows()
                                if row["Favorable"] > self.min_support)

        print("[length:itemsets]: ({}:{})".format(1, len(self.itemsets[1])))
        sys.stdout.flush()

    def create_freq_itemsets(self, superset_max_size=15):
        print("Itemsets creation is in progress, be patient...\n")
        sys.stdout.flush()

        # Creating the first itemsets 
        self.create_initial_itemset()

        # Creating further itemsets with size bigger than 2
        for superset_length in range(2, superset_max_size + 1):

            # Finding candidate itemsets in various lengths up to super_max_size based on preceding itemset
            candidate_superset_freq = defaultdict(int)
            for user_id, user_movies in self.users_movies.items():
                for itemset in self.itemsets[superset_length - 1]:
                    # Check if itemset is a subset of user favorite movies
                    if itemset.issubset(user_movies):
                        # Construct superset with union of current itemset and each of another movies
                        # which user liked separately
                        for other_reviewed_movie in user_movies - itemset: # exclude current movies in itemset first
                            current_superset = itemset | frozenset((other_reviewed_movie,)) # union each remaining itemset
                            # increase the frequency of recent superset which just occurred
                            candidate_superset_freq[current_superset] += 1

            # Checking for frequency of any recent built itemset (candidates) again minimum threshold
            superset = dict([(candidate_superset, candidate_superset_frequency)
                            for candidate_superset, candidate_superset_frequency in candidate_superset_freq.items()
                            if candidate_superset_frequency >= self.min_support])

            print("[length:itemsets]: ({}:{})".format(superset_length, len(superset)))
            sys.stdout.flush()

            if len(superset):
                self.itemsets[superset_length] = superset
            elif len(superset) == 0:
                print("No further exploring.")
                sys.stdout.flush()
                break

        # Itemsets in length 1 are not useful for recommending system so we can drop it
        del self.itemsets[1]

        print('\nItemsets total count: {0}'.format(sum(len(itemsets) for itemsets in self.itemsets.values())))

    def extract_association_rules(self):
        """
        In order to identifying association rules we have to iterate over all itemsets and within each itemset
        pick each member and consider it as conclusion and all others as premises at a time.
        """
        candidate_rules = []
        for itemset_length, itemset_dict in self.itemsets.items():
            for itemset in itemset_dict.keys():
                # selecting each item in itemset and consider it as conclusion
                for conclusion in itemset:
                    # making premise set by excluding conclusion
                    premise = itemset - set((conclusion,))
                    candidate_rules.append((premise, conclusion))

        # Next, we compute the confidence of each of these rules.
        valid_rule = defaultdict(int)
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

        # Calculating confidence level for candidate_rules
        rules_confidence = {
            candidate_rule: valid_rule[candidate_rule] / float(valid_rule[candidate_rule] +
                                                               invalid_rule[candidate_rule])
            for candidate_rule in candidate_rules}

        # Filter out the rules with poor confidence
        self.significant_rules = {rule: confidence for rule, confidence in rules_confidence.items()
                                  if confidence > self.min_confidence}

        print("Among {} candidate rules only which {} of them are significant.".format(len(candidate_rules),
                                                                                       len(self.significant_rules)))

    def get_movie_name(self, movie_id):
        return self.movies.loc[self.movies["MovieID"] == movie_id, 'Title'].values[0]

    def report_associations(self, rule_count=10):
        # Sorting significant rules dictionary based on significant level
        sorted_confidence = sorted(self.significant_rules.items(), key=itemgetter(1), reverse=True)

        for index in range(rule_count):

            (premise, conclusion) = sorted_confidence[index][0]
            premise_names = "\n ".join(self.get_movie_name(movie_id=mov_id) for mov_id in premise)
            conclusion_name = self.get_movie_name(movie_id=conclusion)

            print("Rule rank #{0} (confidence {1:.3f}):".format(index + 1,
                                                                self.significant_rules[(premise, conclusion)]))

            print("If a person recommends:\n {0} \nThey will also recommend: \n {1}".format(premise_names,
                                                                                            conclusion_name))
            print("\n")

    def evaluate_model(self, rule_count=10):
        # Make a test dataset to evaluate model
        test_df = self.ratings_full[self.ratings_full.UserID > self.user_threshold]
        test_fav = test_df[test_df["Favorable"]]

        test_users_movies = dict((test_user_id, frozenset(movies))
                                 for test_user_id, movies in
                                 test_fav.groupby("UserID")["MovieID"])

        candidate_rules = []
        for itemset_length, itemset_dict in self.itemsets.items():
            for itemset in itemset_dict.keys():
                for conclusion in itemset:
                    premise = itemset - set((conclusion,))
                    candidate_rules.append((premise, conclusion))

        # Same evaluation as what we have done in extarcting association rules
        valid_rule = defaultdict(int)
        invalid_rule = defaultdict(int)

        for user, fav_movies in test_users_movies.items():
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

        test_confidence = {candidate_rule: valid_rule[candidate_rule] / float(
            valid_rule[candidate_rule] + invalid_rule[candidate_rule])
                           for candidate_rule in candidate_rules}

        sorted_confidence = sorted(self.significant_rules.items(), key=itemgetter(1), reverse=True)

        for index in range(rule_count):
            (premise, conclusion) = sorted_confidence[index][0]
            premise_names = "\n ".join(self.get_movie_name(movie_id=mov_id) for mov_id in premise)
            conclusion_name = self.get_movie_name(movie_id=conclusion)

            print("Rule rank #{0} \n({1:.3f} confidence )\n({2:.3f} test confidence):".format(index + 1,
                                                                                         self.significant_rules[
                                                                                             (premise, conclusion)],
                                                                                         test_confidence.get(
                                                                                             (premise, conclusion),
                                                                                             -1)))
            print("If a person recommends:\n {0} \nThey will also recommend: \n {1}".format(premise_names,
                                                                                          conclusion_name))
            print("\n")


if __name__ == '__main__':
    engine = MovieRecommendation()
    # Load datasets 
    engine.load_data()

    # Minimizing the dataset size
    engine.data_prep(user_threshold=200)

    # Constructing itemsets
    engine.create_freq_itemsets(superset_max_size=15)

    # Making association rules
    engine.extract_association_rules()

    # Printing reports of extracted rules
    engine.report_associations()

    # Evaluate model
    engine.evaluate_model()
