from typing import List, Tuple


def extract_bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Generate bigrams from a list of tokens.
    """
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    
