import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary_2d(model, X, y, out_path: str, padding=0.5, h=0.02):
    """Assumes X is 2D array (n_samples, 2)."""
    x_min, x_max = X[:,0].min() - padding, X[:,0].max() + padding
    y_min, y_max = X[:,1].min() - padding, X[:,1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor="k")
    plt.title("Decision boundary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

