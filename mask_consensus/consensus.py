def mask_consensus(proposals: list,
                   encrypt_ratio: float,
                   total_params: int) -> list:
    """
    Interleave clients’ proposals, dedupe, take first ρ·N indices.
    """
    m_prime = []
    L = len(proposals[0])
    for i in range(L):
        for p in proposals:
            if i < len(p):
                m_prime.append(p[i])
    unique = list(dict.fromkeys(m_prime))
    k = int(encrypt_ratio * total_params)
    return unique[:k]
