"""Explicit Polish/English category mapping; never infer a severity scale."""
NAMES = ['normal', 'osteophytes', 'parasyndesmophytes', 'syndesmophytes']
ALIASES = {n:i for i,n in enumerate(NAMES)}
ALIASES.update({'osteofity':1, 'parasyndesmofity':2, 'syndesmofity':3})
def category(name):
    key = str(name).strip().lower()
    if key not in ALIASES:
        raise ValueError(f'Unknown morphological category: {name!r}')
    return ALIASES[key]
def probability_order(names):
    if isinstance(names, dict):
        if set(names) != set(range(4)):
            raise ValueError('Checkpoint must contain exactly four indexed classes.')
        names = [names[i] for i in range(4)]
    mapped = [category(n) for n in names]
    if sorted(mapped) != list(range(4)):
        raise ValueError('Checkpoint category mapping is incomplete or duplicated.')
    return [mapped.index(i) for i in range(4)]
