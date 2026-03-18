import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, accuracy_score

class ConvBlock1D(nn.Module):
    """Standard Convolutional Block: Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # padding=1 keeps the length consistent when kernel_size=3
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class WeightedMSELoss(nn.Module):
    def __init__(self, peak_weight=50.0):
        """
        Custom MSE that penalizes errors in the peak regions heavier than the background.
        
        Args:
            peak_weight: The multiplier applied to the loss where the target heatmap > 0.
                         A value of 50 means errors at the true apex are penalized 50x more 
                         than errors in the flat baseline.
        """
        super().__init__()
        self.peak_weight = peak_weight

    def forward(self, y_pred, y_true):
        # 1. Calculate the raw squared error for every pixel
        squared_error = (y_pred - y_true) ** 2
        
        # 2. Build the weight mask
        # Base weight is 1.0 for all pixels.
        # We add the peak_weight scaled by the true heatmap probability.
        # Where y_true is 0 (background), weight remains 1.0.
        # Where y_true is 1 (true apex), weight becomes 1.0 + peak_weight.
        weight_mask = 1.0 + (self.peak_weight * y_true)
        
        # 3. Apply the mask and return the mean
        weighted_error = squared_error * weight_mask
        return torch.mean(weighted_error)

class MultiChannelLoss(nn.Module):
    """
    Loss for multi-channel U-Net (apex heatmap + sigma map + tau map).

    Channel 0 (apex): WeightedMSE — errors at apex pixels penalized peak_weight×.
    Channels 1-2 (sigma, tau): MSE only at apex locations (where target > 0.1),
    weighted by param_weight.
    """
    def __init__(self, peak_weight=50.0, param_weight=1.0):
        super().__init__()
        self.peak_weight = peak_weight
        self.param_weight = param_weight

    def forward(self, y_pred, y_true):
        # Channel 0: apex heatmap — weighted MSE (same as WeightedMSELoss)
        apex_pred = y_pred[:, 0, :]
        apex_true = y_true[:, 0, :]
        apex_se = (apex_pred - apex_true) ** 2
        apex_weight = 1.0 + (self.peak_weight * apex_true)
        apex_loss = torch.mean(apex_se * apex_weight)

        if y_pred.shape[1] < 3:
            return apex_loss

        # Channels 1-2: sigma and tau maps — MSE only at apex locations
        # Mask: where apex heatmap target > 0.1 (near true apex positions)
        apex_mask = (apex_true > 0.1).float()  # (N, L)
        mask_sum = apex_mask.sum()

        if mask_sum < 1.0:
            return apex_loss

        param_loss = 0.0
        for ch in [1, 2]:
            ch_pred = y_pred[:, ch, :]
            ch_true = y_true[:, ch, :]
            ch_se = (ch_pred - ch_true) ** 2
            param_loss += (ch_se * apex_mask).sum() / mask_sum

        return apex_loss + self.param_weight * param_loss


class GCHeatmapUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Build Encoder
        in_c = in_channels
        for feature in features:
            self.encoder.append(ConvBlock1D(in_c, feature))
            in_c = feature

        # Bottleneck (The deepest layer representing the global peak shape)
        self.bottleneck = ConvBlock1D(features[-1], features[-1] * 2)

        # Build Decoder
        for feature in reversed(features):
            # Transposed convolution doubles the sequence length
            self.decoder.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock1D(feature * 2, feature))

        # Final mapping to single channel heatmap [0, 1]
        self.final_conv = nn.Sequential(
            nn.Conv1d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []

        # Forward pass through encoder
        for down_layer in self.encoder:
            x = down_layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse for decoding

        # Forward pass through decoder
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x) # Upsample
            skip_connection = skip_connections[i//2]
            
            # Concatenate skip connection along the channel dimension (dim=1)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i+1](concat_skip) # ConvBlock

        return self.final_conv(x)

def evaluate_and_plot(model, generator, device='cpu'):
    """
    Generates 16 samples, runs inference, extracts peaks, and plots the grid.
    """
    # 1. Generate new test data
    X_test_np, y_true_np = generator.generate_batch(num_samples=16)
    
    # 2. Format for PyTorch (N, C, L)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).permute(0, 2, 1).to(device)
    
    # 3. Run Inference
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    with torch.no_grad():
        predictions = model(X_test_tensor)
        
    # 4. Convert back to NumPy (N, L, C) for plotting
    predictions_np = predictions.cpu().permute(0, 2, 1).numpy()
    
    # 5. Set up the matplotlib grid
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flatten()
    
    for i in range(16):
        ax = axes[i]
        signal = X_test_np[i, :, 0]
        heatmap_pred = predictions_np[i, :, 0]
        
        # --- The Key Extraction Step ---
        # We search the predicted heatmap for peaks.
        # Height filters out low-probability noise; distance prevents double-counting a wide blob.
        apex_indices, _ = find_peaks(heatmap_pred, height=0.15, distance=10)
        
        # Plot the original noisy GC signal
        ax.plot(signal, label='Raw Signal', color='#1f77b4', linewidth=1.5)
        
        # Plot the model's predicted heatmap
        ax.plot(heatmap_pred, label='Pred Heatmap', color='#ff7f0e', linestyle='--', alpha=0.8)
        
        # Scatter plot red markers at the inferred apex locations
        if len(apex_indices) > 0:
            ax.plot(apex_indices, signal[apex_indices], 'rx', 
                    markersize=10, markeredgewidth=2, label='Inferred Apex')
            
        # Clean up the plot aesthetics
        ax.set_title(f"Sample {i+1} | Peaks Found: {len(apex_indices)}")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Only show legend on the first subplot to keep the grid clean
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            
    plt.tight_layout()
    plt.show()

def evaluate_classification_metrics(model, generator, num_samples=1000, device='cpu',
                                     max_count_label=4):
    """
    Evaluates the 1D U-Net as a multi-class peak-count classifier and plots
    representative examples grouped by correct, over-predicted, and under-predicted.

    Each 256-pt window is classified by its exact peak count (capped at
    max_count_label, shown as "N+").

    Parameters
    ----------
    generator : object
        Must expose generate_batch(num_samples) → (X, Y) where X and Y are
        (N, 256, 1) float arrays.
    max_count_label : int
        Peak counts above this are clamped to this label (displayed as "N+").
    """
    from collections import defaultdict
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(f"Generating {num_samples} test samples...")
    X_test_np, y_true_np = generator.generate_batch(num_samples=num_samples)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).permute(0, 2, 1).to(device)

    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)

    predictions_np = predictions_tensor.cpu().permute(0, 2, 1).numpy()

    y_true_counts = []
    y_pred_counts = []

    for i in range(num_samples):
        true_peaks, _ = find_peaks(y_true_np[i, :, 0], height=0.5)
        n_true = min(len(true_peaks), max_count_label)
        y_true_counts.append(n_true)

        pred_peaks, _ = find_peaks(predictions_np[i, :, 0], height=0.15, distance=10)
        n_pred = min(len(pred_peaks), max_count_label)
        y_pred_counts.append(n_pred)

    labels = list(range(max_count_label + 1))
    label_names = [f'{l}+' if l == max_count_label else str(l) for l in labels]

    acc = accuracy_score(y_true_counts, y_pred_counts)
    n_correct = sum(t == p for t, p in zip(y_true_counts, y_pred_counts))
    n_over    = sum(p > t   for t, p in zip(y_true_counts, y_pred_counts))
    n_under   = sum(p < t   for t, p in zip(y_true_counts, y_pred_counts))
    cm        = confusion_matrix(y_true_counts, y_pred_counts, labels=labels)

    print('-' * 50)
    print(f'Accuracy:         {acc:.4f}  ({n_correct}/{num_samples} exact)')
    print(f'Over-prediction:  {n_over:5d}  ({100*n_over/num_samples:.1f}%)')
    print(f'Under-prediction: {n_under:5d}  ({100*n_under/num_samples:.1f}%)')
    print()
    print('Classification Report (by peak count):')
    print(classification_report(y_true_counts, y_pred_counts,
                                 labels=labels, target_names=label_names,
                                 zero_division=0))
    print('Confusion Matrix (rows=true, cols=predicted):')
    header = '      ' + ''.join(f'{n:>7}' for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f'{label_names[i]:>5} ' + ''.join(f'{v:>7}' for v in row))
    print('-' * 50)

    # --- Confusion matrix heatmap ---
    n_labels = len(label_names)
    fig_cm, ax_cm = plt.subplots(figsize=(max(5, n_labels + 1), max(4, n_labels)))
    im = ax_cm.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax_cm)
    ax_cm.set_xticks(range(n_labels))
    ax_cm.set_yticks(range(n_labels))
    ax_cm.set_xticklabels(label_names)
    ax_cm.set_yticklabels(label_names)
    ax_cm.set_xlabel('Predicted Peak Count', fontsize=11)
    ax_cm.set_ylabel('True Peak Count', fontsize=11)
    ax_cm.set_title('Confusion Matrix — Peak Count', fontsize=12)
    thresh = cm.max() * 0.6
    for i in range(n_labels):
        for j in range(n_labels):
            color = 'white' if cm[i, j] > thresh else 'black'
            weight = 'bold' if i == j else 'normal'
            ax_cm.text(j, i, str(cm[i, j]),
                       ha='center', va='center',
                       color=color, fontsize=11, fontweight=weight)
    fig_cm.tight_layout()
    plt.show()

    # --- Example cases: correct / over / under ---
    pair_to_indices = defaultdict(list)
    for i, (t, p) in enumerate(zip(y_true_counts, y_pred_counts)):
        pair_to_indices[(t, p)].append(i)

    def _pick_pairs(pairs, n=2):
        selected = []
        for pair in pairs:
            if pair in pair_to_indices:
                selected.append((pair, pair_to_indices[pair][0]))
            if len(selected) == n:
                break
        return selected

    correct_pairs = sorted([k for k in pair_to_indices if k[0] == k[1]],
                            key=lambda x: x[0], reverse=True)
    over_pairs    = sorted([k for k in pair_to_indices if k[1] > k[0]],
                            key=lambda x: (x[1] - x[0], len(pair_to_indices[x])), reverse=True)
    under_pairs   = sorted([k for k in pair_to_indices if k[1] < k[0]],
                            key=lambda x: (x[0] - x[1], len(pair_to_indices[x])), reverse=True)

    cases = []
    for pair, idx in _pick_pairs(correct_pairs):
        n = len(pair_to_indices[pair])
        cases.append((f'Correct (True={pair[0]}, Pred={pair[1]}, n={n})', 'correct', idx))
    for pair, idx in _pick_pairs(over_pairs):
        n = len(pair_to_indices[pair])
        cases.append((f'Over-pred (True={pair[0]}, Pred={pair[1]}, n={n})', 'over', idx))
    for pair, idx in _pick_pairs(under_pairs):
        n = len(pair_to_indices[pair])
        cases.append((f'Under-pred (True={pair[0]}, Pred={pair[1]}, n={n})', 'under', idx))

    if cases:
        ncols = min(3, len(cases))
        nrows = (len(cases) + ncols - 1) // ncols
        fig_ex, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                    squeeze=False)
        kind_colors = {'correct': '#2ca02c', 'over': '#d62728', 'under': '#ff7f0e'}
        x_axis = range(256)

        for k, (title, kind, idx) in enumerate(cases):
            ax = axes[k // ncols][k % ncols]
            sig       = X_test_np[idx, :, 0]
            hmap_pred = predictions_np[idx, :, 0]
            hmap_true = y_true_np[idx, :, 0]
            pp, _ = find_peaks(hmap_pred, height=0.15, distance=10)
            tp, _ = find_peaks(hmap_true, height=0.5)

            ax.plot(sig,       color='#1f77b4', linewidth=1,   label='Signal')
            ax.plot(hmap_true, color='green',   linewidth=0.8, linestyle=':', alpha=0.7, label='True heatmap')
            ax.plot(hmap_pred, color='red',     linewidth=1.2, linestyle='--', alpha=0.8, label='Pred heatmap')
            if len(tp) > 0:
                ax.plot(tp, sig[tp], '^', color='green', markersize=7, label='True apex')
            if len(pp) > 0:
                ax.plot(pp, sig[pp], 'rx', markersize=10, markeredgewidth=2, label='Pred apex')

            ax.set_title(title, fontsize=8, color=kind_colors[kind])
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.4)

        for k in range(len(cases), nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)

        fig_ex.suptitle('Peak Count Prediction Examples', fontsize=11)
        fig_ex.tight_layout()
        plt.show()