import matplotlib.pyplot as plt
from IPython import display
from pathlib import Path

# Plotting function
plt.ion()


def plot(scores, mean_scores, model_number="00"):
    """
    Plot the scores

    :param model_number: - The model number to save the plot
    :param scores: - The scores
    :param mean_scores: - The mean scores
    """

    # Clear the display
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    # Plot the data
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    # Show the plot
    plt.show(block=False)
    plt.pause(.1)

    # Save the plot
    plot_path = Path("../assets/plots")
    plot_path.mkdir(parents=True, exist_ok=True)

    # Save the plot by replacing the last plot
    plt.savefig(plot_path / f"plot_model{model_number}.png")
