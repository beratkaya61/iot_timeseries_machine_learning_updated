import matplotlib.pyplot as plt

def plot_predictions(true, pred, title="Prediction vs Actual"):
    plt.figure(figsize=(12, 6))
    plt.plot(true, label='True')
    plt.plot(pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
