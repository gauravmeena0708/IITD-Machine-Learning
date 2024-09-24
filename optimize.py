import random
import numpy as np
import itertools
from logger import Logger
logger = Logger(filename="Optimize", logging="TXT")

def pso(objective_function, lower_bound, upper_bound, n_particles, n_dimensions, max_iter, w=0.9, c1=0.01, c2=0.1):
    possible_combinations = list(itertools.product(*possible_values))
    particles = np.array(possible_combinations)

    if len(particles) < n_particles:
        additional_particles = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_particles - len(particles), n_dimensions))
        particles = np.vstack((particles, additional_particles))
    else:
        n_particles = len(particles)
    personal_best_positions = particles.copy()
    global_best_position = particles[np.argmin([objective_function(p) for p in particles])]

    # Initialize the velocities
    velocities = np.zeros((n_particles, n_dimensions))
    for i in range(max_iter):
        r1 = np.random.rand(n_particles, n_dimensions)
        r2 = np.random.rand(n_particles, n_dimensions)
        velocities = w * velocities + c1 * r1 * (personal_best_positions - particles) + c2 * r2 * (global_best_position - particles)
        particles = particles + velocities
        particles = np.clip(particles, lower_bound, upper_bound)

        for j in range(n_particles):
            if objective_function(particles[j]) < objective_function(personal_best_positions[j]):
                personal_best_positions[j] = particles[j].copy()
            if objective_function(personal_best_positions[j]) < objective_function(global_best_position):
                global_best_position = personal_best_positions[j].copy()

    return global_best_position

# Random Search with logging through combinations of possible values
def random_search(evaluation_function, possible_values, *args, n_iterations=1000):
    best_solution = None
    best_value = float('inf')

    # Generate all possible combinations of parameters
    possible_combinations = list(itertools.product(*possible_values))

    # Limit iterations to the number of possible combinations if it's less than n_iterations
    n_iterations = min(n_iterations, len(possible_combinations))

    for i in range(n_iterations):
        candidate = possible_combinations[i]
        candidate_value = evaluation_function(candidate, *args)
        if candidate_value < best_value:
            best_value = candidate_value
            best_solution = candidate
            logger.log(f"Random: Iter: {i+1} - Candidate: {candidate}, Evaluation Value: {candidate_value}, Best Solution: {best_solution}, Best Evaluation Value: {best_value}")
    
    return best_solution, best_value

# Example usage
if __name__ == "__main__":
    # Define the possible values for each parameter
    possible_values = [
        [0.0001, 0.001, 0.01, 0.1],  # Possible values for rate
        [0.5, 0.6, 0.7, 0.8, 0.9],  # Possible values for beta1
        [16, 32, 64, 128],  # Possible values for batch_size
        [10, 15, 20]  # Possible values for test
    ]

    # Calculate bounds based on possible values
    lower_bound = [min(values) for values in possible_values]
    upper_bound = [max(values) for values in possible_values]
    n_particles = 10
    n_dimensions = len(possible_values)
    max_iter = 10

    # Define your evaluation function
    def evaluate_hyperparameters(params):
        # Example evaluation function, replace with your actual function
        rate, beta1, batch_size, test = params
        # Assuming your evaluation function returns some metric based on the hyperparameters
        return rate + beta1 + batch_size + test * test


    best_solution2, best_value2 = random_search(evaluate_hyperparameters, possible_values)
    logger.log(f"Random: Best Solution{best_solution2} and Best_value:{best_value2}")
    best_solution1 = pso(evaluate_hyperparameters, lower_bound, upper_bound, n_particles, n_dimensions, max_iter)
    best_value1 = evaluate_hyperparameters(best_solution1)
    logger.log(f"PSO: Best Solution{best_solution1} and Best_value:{best_value1}")
    if best_value1<best_value2:
        best_value = best_value1
        best_solution = best_solution1
    else:
        best_value = best_value2
        best_solution = best_solution2

    logger.log(f"\nFinal Best Solution: {best_solution}")
    logger.log(f"Final Best Evaluation Value: {best_value}")
