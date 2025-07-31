import yaml
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.resnet18           import get_resnet18
from clients.client            import Client
from server.server             import Server
from encryption.encryptor      import init_context
from mask_consensus.consensus  import mask_consensus
from utils.helpers             import vectorize_model, update_model

def load_mnist(batch_size):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def main():
    cfg      = yaml.safe_load(open('config.yaml'))
    R, K, B  = cfg['rounds'], cfg['num_clients'], cfg['batch_size']
    RATIO    = cfg['encrypt_ratio']

    client_ctx, server_ctx = init_context()
    base_model = get_resnet18(num_classes=10, input_channels=1)
    server     = Server()
    server.global_vector = vectorize_model(base_model)

    loaders = [load_mnist(B) for _ in range(K)]
    clients = [
        Client(i,
               get_resnet18(num_classes=10, input_channels=1),
               loaders[i],
               client_ctx,
               server_ctx,
               RATIO)
        for i in range(K)
    ]

    # Save one‐batch gradient for IDLG
    grad = clients[0].get_attack_gradient()
    torch.save(grad, 'experiments/attack_gradient.pt')
    print("🔒 attack_gradient.pt saved")

    for r in range(R):
        print(f"\n--- Round {r+1}/{R} ---")
        proposals, plains = [], []

        for c in clients:
            update_model(c.model, server.global_vector)
            c.local_train(epochs=1)
            proposals.append(c.get_mask_proposal(server.global_vector))
            plains.append(c.get_plain_vector())

        N = server.global_vector.numel()
        mask = list(range(N)) if RATIO == 1.0 else mask_consensus(proposals, RATIO, N)

        enc_chunks = [c.encrypt_updates(mask) for c in clients]
        agg_plain, agg_enc_chunks = server.aggregate(plains, enc_chunks, mask)

        dec_list  = clients[0].decrypt_aggregate(agg_enc_chunks)
        dec_slice = torch.tensor(dec_list, dtype=agg_plain.dtype, device=agg_plain.device)

        new_global = agg_plain.clone()
        new_global[mask] = dec_slice
        server.global_vector = new_global

    print("✅ Federated training complete.")

if __name__ == '__main__':
    main()
