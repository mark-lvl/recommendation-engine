# recommendation-engine
Implementing Apriori algorithm for affinity analysis on movie dataset and extracting association rules as movie recommendation.

Affinity analysis is the task of determining when objects are used in similar ways.
Affinity analysis is usually much more exploratory than classification. We often don't have the complete dataset we expect for many classification tasks.

The classic algorithm for affinity analysis is called the Apriori algorithm. It addresses the exponential problem of creating sets of items that occur frequently within a database, called frequent itemsets. Once these frequent itemsets are discovered, creating association rules is straightforward.

The intuition behind Apriori is both simple and clever. First, we ensure that a rule has sufficient support within the dataset. Defining a minimum support level is the key parameter for Apriori. To build a frequent itemset, for an itemset (A, B) to have a support of at least 30, both A and B must occur at least 30 times in the database. This property extends to larger sets as well. For an itemset (A, B, C, D) to be considered frequent, the set (A, B, C) must also be frequent (as must D).

These frequent itemsets can be built up and possible itemsets that are not frequent (of which there are many) will never be tested. This saves significant time in testing new rules.