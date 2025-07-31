import yaml
import time
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.resnet18           import get_resnet18
from clients.client            import Client
from server.server             import Server
from encryption.encryptor      import init_context
from mask_consensus.consensus  import mask_consensus
from utils.helpers             import vectorize_model, update_model

def load_mnist(batch_size, train=True):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST('./data', train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=train)

def eval_model(vector, loader):
    model = get_resnet18(num_classes=10, input_channels=1).eval()
    torch.nn.utils.vector_to_parameters(vector, model.parameters())
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            _, p = out.max(1)
            correct += (p == y).sum().item()
            total   += y.size(0)
    return correct / total

def main():
    cfg       = yaml.safe_load(open('config.yaml'))
    R, K, B   = cfg['rounds'], cfg['num_clients'], cfg['batch_size']
    INF_RATIO = cfg['inflation_ratio']

    ratios     = [0.0, 0.05, 0.1, 0.2, 1.0]
    test_loader = load_mnist(B, train=False)

    records = []
    for rto in ratios:
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
                   rto)
            for i in range(K)
        ]

        start = time.time()
        for _ in range(R):
            props, plains = [], []
            for c in clients:
                update_model(c.model, server.global_vector)
                c.local_train(epochs=1)
                props.append(c.get_mask_proposal(server.global_vector))
                plains.append(c.get_plain_vector())

            N = server.global_vector.numel()
            mask = list(range(N)) if rto == 1.0 else mask_consensus(props, rto, N)

            enc_chunks = [c.encrypt_updates(mask) for c in clients]
            agg_p, agg_ct = server.aggregate(plains, enc_chunks, mask)
            dec_list = clients[0].decrypt_aggregate(agg_ct)
            dec_slice = torch.tensor(dec_list, dtype=agg_p.dtype, device=agg_p.device)

            new_g = agg_p.clone()
            new_g[mask] = dec_slice
            server.global_vector = new_g

        elapsed = time.time() - start
        comm_mult = round((1 - rto + rto * INF_RATIO), 2)
        acc       = round(eval_model(server.global_vector, test_loader), 4)
        records.append({
            'encrypt_ratio':   rto,
            'comm_multiplier': comm_mult,
            'time_sec':        round(elapsed, 2),
            'test_accuracy':   acc
        })

    df = pd.DataFrame(records)
    df.to_csv('experiments/metrics.csv', index=False)
    print("✅ Saved experiments/metrics.csv")
    print(df)

if __name__ == '__main__':
    main()
