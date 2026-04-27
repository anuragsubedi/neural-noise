import matplotlib.pyplot as plt

steps = [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 3999]
train_loss = [6.2411, 3.9064, 3.2280, 2.9312, 2.4722, 2.1481, 1.9131, 1.7861, 1.6246, 1.5259, 1.4366]
val_loss = [6.2465, 3.8834, 3.1950, 2.8993, 2.4361, 2.1548, 2.0052, 1.9203, 1.8392, 1.8311, 1.8514]
train_ppl = [513.4016, 49.7172, 25.2283, 18.7503, 11.8484, 8.5690, 6.7737, 5.9659, 5.0763, 4.5991, 4.2063]
val_ppl = [516.1938, 48.5906, 24.4096, 18.1612, 11.4287, 8.6259, 7.4279, 6.8232, 6.2914, 6.2405, 6.3690]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Cross-Entropy Loss
ax1.plot(steps, train_loss, label='Train Loss', marker='o', linewidth=2, color='#1f77b4')
ax1.plot(steps, val_loss, label='Validation Loss', marker='s', linewidth=2, color='#ff7f0e')
ax1.set_title('Cross-Entropy Loss vs Steps', fontsize=14, pad=15)
ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=11)

# Plot Perplexity
ax2.plot(steps, train_ppl, label='Train Perplexity', marker='o', linewidth=2, color='#2ca02c')
ax2.plot(steps, val_ppl, label='Validation Perplexity', marker='s', linewidth=2, color='#d62728')
ax2.set_title('Perplexity vs Steps (Log Scale)', fontsize=14, pad=15)
ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Perplexity', fontsize=12)
ax2.set_yscale('log')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=11)

# Style adjustments
fig.suptitle('MicroMusicGPT Training Convergence (Colab A100)', fontsize=16, y=1.05)
plt.tight_layout()

# Save the plot
plt.savefig('498_docs/training_metrics.png', dpi=300, bbox_inches='tight')
print("Successfully generated and saved metrics chart to 498_docs/training_metrics.png")
