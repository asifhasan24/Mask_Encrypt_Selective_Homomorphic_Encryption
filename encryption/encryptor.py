import tenseal as ts

def init_context():
    """
    Returns:
      client_ctx: TenSEALContext WITH secret key (for decryption)
      server_ctx: TenSEALContext PUBLIC ONLY (for encryption & aggregation)
    """
    # 1) set a known poly_modulus_degree
    degree = 16384
    client_ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=degree,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    client_ctx.global_scale = 2**40
    client_ctx.generate_galois_keys()
    client_ctx.generate_relin_keys()

    # 2) public‐only copy
    server_ctx = client_ctx.copy()
    server_ctx.make_context_public()
    # attach degree so we can chunk correctly
    server_ctx._poly_modulus_degree = degree
    return client_ctx, server_ctx

def encrypt_vector(pt_vector, server_context):
    """
    Chunk the plaintext vector into slices that fit into one ciphertext.
    Returns a list of CKKSVector.
    """
    data = pt_vector.tolist() if not isinstance(pt_vector, list) else pt_vector
    # max slots = poly_modulus_degree//2
    max_slots = server_context._poly_modulus_degree // 2
    chunks = []
    for i in range(0, len(data), max_slots):
        slice_ = data[i:i+max_slots]
        chunks.append(ts.ckks_vector(server_context, slice_))
    return chunks

def decrypt_vector(ct_chunks, client_context):
    """
    Decrypt a list of CKKSVector, concatenating their plaintexts.
    """
    sk = client_context.secret_key()
    result = []
    for ct in ct_chunks:
        result.extend(ct.decrypt(secret_key=sk))
    return result
