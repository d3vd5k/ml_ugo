import matplotlib.pyplot as plt

def plot_datasets(X_orig, y_orig, X_resampled, y_resampled):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(X_orig[y_orig==0][:, 0], X_orig[y_orig==0][:, 1], label='Class 0', alpha=0.6)
    ax1.scatter(X_orig[y_orig==1][:, 0], X_orig[y_orig==1][:, 1], label='Class 1', alpha=0.6)
    ax1.set_title('Original Dataset')
    ax1.legend()
    
    n_orig = len(X_orig)
    ax2.scatter(X_resampled[:n_orig][y_resampled[:n_orig]==0][:, 0], X_resampled[:n_orig][y_resampled[:n_orig]==0][:, 1], label='Class 0', alpha=0.6)
    ax2.scatter(X_resampled[:n_orig][y_resampled[:n_orig]==1][:, 0], X_resampled[:n_orig][y_resampled[:n_orig]==1][:, 1], label='Class 1', alpha=0.6)
    ax2.scatter(X_resampled[n_orig:][:, 0], X_resampled[n_orig:][:, 1], marker='+', color='green', label='Synthetic Data', s=50)
    ax2.set_title('UGO Resampled Dataset')
    ax2.legend()
    
    plt.show()