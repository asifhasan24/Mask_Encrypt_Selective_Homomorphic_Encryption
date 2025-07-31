import torch

class Server:
    # ... __init__ stays the same ...

    def aggregate(self,
                  plain_vecs: list,
                  enc_vecs_chunks: list,
                  mask_indices: list):
        """
        enc_vecs_chunks: list (per client) of lists of CKKSVector chunks
        """
        num = len(plain_vecs)
        N   = plain_vecs[0].numel()

        # 1) average plaintext parts
        masked = []
        for v in plain_vecs:
            tmp = v.clone()
            tmp[mask_indices] = 0.0
            masked.append(tmp)
        agg_plain = torch.stack(masked,0).mean(0)

        # 2) homomorphic sum/avg chunk‐wise
        num_chunks = len(enc_vecs_chunks[0])
        agg_enc = []
        for ci in range(num_chunks):
            ct_sum = enc_vecs_chunks[0][ci]
            for c in enc_vecs_chunks[1:]:
                ct_sum += c[ci]
            agg_enc.append(ct_sum * (1.0/num))

        return agg_plain, agg_enc
