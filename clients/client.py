import torch
from mask_selection.gradient_mask import gradient_guided_mask
from encryption.encryptor       import encrypt_vector, decrypt_vector

class Client:
    def __init__(self,
                 cid: int,
                 model: torch.nn.Module,
                 loader: torch.utils.data.DataLoader,
                 client_ctx,
                 server_ctx,
                 encrypt_ratio: float):
        """
        cid           : client ID
        model         : your PyTorch model
        loader        : local DataLoader
        client_ctx    : TenSEALContext with secret key
        server_ctx    : TenSEALContext public only
        encrypt_ratio : ρ, fraction to encrypt each round
        """
        self.id         = cid
        self.model      = model
        self.loader     = loader
        self.client_ctx = client_ctx
        self.server_ctx = server_ctx
        self.ratio      = encrypt_ratio
        self.opt        = torch.optim.SGD(self.model.parameters(),
                                          lr=0.01)

    def local_train(self, epochs: int = 1):
        """Standard local training with SGD."""
        self.model.train()
        for _ in range(epochs):
            for x, y in self.loader:
                self.opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(x), y)
                loss.backward()
                self.opt.step()

    def get_mask_proposal(self, global_vector: torch.Tensor) -> list:
        """
        Compute gradient-guided mask proposal:
        - w_t: current local weights
        - w_prev: exposed global vector
        - grad_vec: gradient on one mini-batch
        """
        # Flatten current & previous weights
        w_t    = torch.nn.utils.parameters_to_vector(
                     self.model.parameters()).detach()
        w_prev = global_vector.clone()

        # Compute single-batch gradient
        x, y   = next(iter(self.loader))
        self.opt.zero_grad()
        out    = self.model(x)
        loss   = torch.nn.functional.cross_entropy(out, y)
        grads  = torch.autograd.grad(loss, self.model.parameters())
        grad_vec = torch.cat([g.detach().flatten() for g in grads])

        # Return top-ρ·N indices
        return gradient_guided_mask(w_t, w_prev, grad_vec, self.ratio)

    def get_plain_vector(self) -> torch.Tensor:
        """Flatten the entire model into a 1D tensor."""
        return torch.nn.utils.parameters_to_vector(
                   self.model.parameters()).detach()

    def encrypt_updates(self, mask_indices: list):
        """
        Encrypt only the selected indices.
        Returns: list of CKKSVector chunks of the selected slice.
        Works for both selective (mask_indices ⊂ all) and full HE (mask_indices == all).
        """
        vec      = self.get_plain_vector()
        selected = vec[mask_indices].tolist()
        return encrypt_vector(selected, self.server_ctx)

    def decrypt_aggregate(self, enc_agg_chunks):
        """
        Decrypt a list of CKKSVector chunks (the aggregated slice).
        Returns a Python list of floats.
        """
        return decrypt_vector(enc_agg_chunks, self.client_ctx)

    def get_attack_gradient(self) -> torch.Tensor:
        """
        Expose a single-batch gradient vector for the IDLG inversion attack.
        """
        x, y = next(iter(self.loader))
        self.opt.zero_grad()
        out  = self.model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        grads= torch.autograd.grad(loss, self.model.parameters())
        return torch.cat([g.detach().flatten() for g in grads])
