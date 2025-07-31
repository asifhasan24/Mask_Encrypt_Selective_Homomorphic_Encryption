import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from models.resnet18 import get_resnet18

# ----------------
# IDLG inversion
# ----------------
def idlg_attack(grad_path: str, iters: int = 300, lr: float = 0.1):
    target_grad = torch.load(grad_path)
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    model       = get_resnet18(num_classes=10, input_channels=1).to(device).eval()
    model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False).to(device)

    dummy       = torch.randn(1,1,28,28, device=device, requires_grad=True)
    label       = torch.zeros(1, dtype=torch.long, device=device)
    optimizer   = torch.optim.LBFGS([dummy], lr=lr)
    target_grad = target_grad.to(device)

    def closure():
        optimizer.zero_grad()
        out     = model(dummy)
        loss_ce = F.cross_entropy(out, label)
        grads   = torch.autograd.grad(loss_ce, model.parameters(), create_graph=True)
        grad_vec= torch.cat([g.flatten() for g in grads])
        loss_mse= F.mse_loss(grad_vec, target_grad)
        loss_mse.backward()
        return loss_mse

    for i in range(iters):
        optimizer.step(closure)
        if i % 50 == 0:
            print(f"Iter {i}/{iters}, MSE={closure().item():.4f}")

    return dummy.detach().cpu().squeeze()


# ----------------
# Main evaluation
# ----------------
def main():
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    # 1) Load & print metrics
    df = pd.read_csv('experiments/metrics.csv')
    print("\n=== Benchmark Metrics ===")
    print(df.to_string(index=False))

    # 2) Communication overhead plot
    plt.figure()
    plt.plot(df['encrypt_ratio'], df['comm_multiplier'], '-o')
    plt.xlabel('Encryption Ratio')
    plt.ylabel('Comm Multiplier (×)')
    plt.title('Communication Overhead vs Encryption Ratio')
    plt.grid(True)
    plt.savefig('plots/comm_overhead.png')
    plt.show()

    # 3) Training time plot
    plt.figure()
    plt.plot(df['encrypt_ratio'], df['time_sec'], '-s', color='C1')
    plt.xlabel('Encryption Ratio')
    plt.ylabel('Wall-Time (s)')
    plt.title('Training Time vs Encryption Ratio')
    plt.grid(True)
    plt.savefig('plots/training_time.png')
    plt.show()

    # 4) Test accuracy plot
    plt.figure()
    plt.plot(df['encrypt_ratio'], df['test_accuracy'], '-^', color='C2')
    plt.xlabel('Encryption Ratio')
    plt.ylabel('Test Accuracy')
    plt.title('Final Test Accuracy vs Encryption Ratio')
    plt.grid(True)
    plt.savefig('plots/test_accuracy.png')
    plt.show()

    # 5) IDLG inversion attack & save
    print("\n=== IDLG Inversion Attack ===")
    recon = idlg_attack('experiments/attack_gradient.pt')
    plt.figure(figsize=(4,4))
    plt.imshow(recon, cmap='gray')
    plt.title("IDLG Reconstructed MNIST")
    plt.axis('off')
    plt.savefig('plots/idlg_reconstruction.png')
    plt.show()


if __name__ == '__main__':
    main()
