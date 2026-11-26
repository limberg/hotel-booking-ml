import numpy as np


def calculate_average(values):
    """Calculate the average of a list of values."""
    result = np.mean(values)
    return result


numbers = [10, 20, 30, 40, 50]

average = calculate_average(numbers)

print(f"Average: {average}")
