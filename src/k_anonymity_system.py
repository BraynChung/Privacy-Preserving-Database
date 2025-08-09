import pandas as pd

class KAnonymitySystem:
    """
    Simple K-Anonymity implementation for performance assessment.
    This version generalizes quasi-identifiers to achieve k-anonymity.
    """
    def __init__(self, k=5):
        self.k = k

    def anonymize(self, df: pd.DataFrame, quasi_identifiers: list):
        # For demonstration, generalize by grouping and suppressing rare groups
        grouped = df.groupby(quasi_identifiers)
        sizes = grouped.size().reset_index(name='count')
        mask = sizes['count'] >= self.k
        valid_groups = sizes[mask][quasi_identifiers]
        # Only keep rows in groups of size >= k
        merged = df.merge(valid_groups, on=quasi_identifiers, how='inner')
        # Optionally, suppress or generalize further here
        return merged