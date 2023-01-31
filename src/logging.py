"""
Functions for logging data/saving to files.
"""

# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from pathlib import Path


# =============================================================================
#  FUNCTIONS
# =============================================================================

def log_results(results: dict) -> None:
    """ Log results dictionary to a text file.

    Args:
        results (dict) : Results to log (keys = model, values = metrics)

    Returns:
        None
    """
    # Create the file
    with open("results.txt", "w") as f:
        f.write("Results\n-------\n")

        # Log results to file sequentially
        for key, value in results.items():
            f.write(f"{key} model: Accuracy = {value:.4f}\n")

