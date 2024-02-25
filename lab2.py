from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

data = [['milk', 'bread', 'butter'],
        ['milk', 'bread'],
        ['milk', 'butter'],
        ['milk', 'eggs'],
        ['bread', 'butter'],
        ['milk', 'eggs'],
        ['milk', 'bread', 'butter'],
        ['milk', 'bread'],
        ['milk', 'butter'],
        ['milk', 'eggs', 'bread']]


encoder = TransactionEncoder()

one_hot_encoded = encoder.fit(data).transform(data)

df = pd.DataFrame(one_hot_encoded, columns=encoder.columns_)


frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)
