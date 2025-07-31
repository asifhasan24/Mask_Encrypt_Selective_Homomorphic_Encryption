# MaskCrypt Project

This repository contains an end-to-end implementation of **MaskCrypt: Federated Learning With Selective Homomorphic Encryption** on MNIST, featuring:

* Selective homomorphic encryption (CKKS via TenSEAL) with a gradient-guided mask.
* Mask consensus across clients.
* Dynamic evaluation: communication overhead, training time, test accuracy.
* IDLG inversion attack (reconstruct input from gradients).

---

## 📁 Folder Structure

```
maskcrypt_project/
├── config.yaml               # Hyperparameters and settings
├── requirements.txt          # Python dependencies
├── models/                   # Model definitions
│   └── resnet18.py
├── encryption/               # Homomorphic encryption utils
│   └── encryptor.py
├── mask_selection/           # Gradient-guided mask selection
│   └── gradient_mask.py
├── mask_consensus/           # Mask consensus algorithm
│   └── consensus.py
├── clients/                  # Client-side training & encryption
│   └── client.py
├── server/                   # Server-side aggregation
│   └── server.py
├── utils/                    # Misc. helpers
│   └── helpers.py
└── experiments/              # Experiment scripts
    ├── train_mnist.py        # Federated training + save gradient
    ├── benchmark_mnist.py    # Benchmark across encryption ratios
    └── evaluator.py          # Dynamic evaluation & IDLG attack
```

---

## ⚙️ Installation

1. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\\Scripts\\activate
   ```
2. **Install the required Python packages**:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Run Order

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
2. **Federated Training & Save Attack Gradient**

   ```bash
   python -m experiments.train_mnist
   ```

   * Trains MaskCrypt on MNIST for configured rounds.
   * Saves one-batch gradient for IDLG to `experiments/attack_gradient.pt`.
3. **Benchmark Across Encryption Ratios**

   ```bash
   python -m experiments.benchmark_mnist
   ```

   * Runs federated rounds for multiple encryption ratios `[0.0,0.05,0.1,0.2,1.0]`.
   * Outputs metrics CSV to `experiments/metrics.csv`.
4. **Dynamic Evaluation & IDLG Inversion Attack**

   ```bash
   python -m experiments.evaluator
   ```

   * Displays metrics table.
   * Plots: communication overhead, training time, test accuracy.
   * Shows the IDLG-reconstructed image.

---

## 🔧 Configuration

Modify `config.yaml` to adjust:

* `encrypt_ratio` (float): fraction of weights encrypted per round.
* `rounds` (int): number of federated rounds.
* `num_clients` (int): number of simulated clients.
* `batch_size` (int): mini-batch size for local training.
* `learning_rate` (float): learning rate for SGD.
* `inflation_ratio` (float): CKKS ciphertext inflation used in benchmark.

---

## 📄 License

MIT License. Feel free to use and adapt for your research.
