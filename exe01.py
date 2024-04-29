import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GeneticAlgorithmProblem1:
    def __init__(self):
        self.population_size = 60
        self.population = []
        self.num_generations = 10
        self.num_children = 30
        self.children = []
        self.mutation_rate = 0.1
        self.all_populations = []

    def evaluate_individual(self, x):
        return 837.9658 - (np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def create_population(self):
        for _ in range(self.population_size):
            individual = [random.uniform(-500, 500) for _ in range(2)]
            individual.append(self.evaluate_individual(individual))
            self.population.append(individual)

    def select_parents(self):
        fitness_total = sum(individual[2] for individual in self.population)
        weights = [individual[2] / fitness_total for individual in self.population]
        parent1 = random.choices(range(len(self.population)), weights=weights)[0]
        parent2 = random.choices(range(len(self.population)), weights=weights)[0]
        return parent1, parent2

    def mutate(self, child):
        for i in range(2):
            if random.random() < self.mutation_rate:
                child[i] = random.uniform(-500, 500)
        return child

    def discard_population(self):
        self.population.sort(key=lambda x: x[2])
        del self.population[:-self.num_children]

    def reproduce(self):
        for _ in range(self.num_children // 2):
            parent1, parent2 = self.select_parents()
            x1, y1 = self.population[parent1][:2]
            x2, y2 = self.population[parent2][:2]

            child1 = [x1, y2]
            child2 = [x2, y1]

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            child1.append(self.evaluate_individual(child1))
            child2.append(self.evaluate_individual(child2))

            self.children.append(child1)
            self.children.append(child2)

    def init_execution(self):
        self.create_population()
        for generation in range(1, self.num_generations + 1):
            self.reproduce()
            self.population.extend(self.children)
            self.discard_population()
            self.all_populations.append(self.population.copy())
            self.print_generation_info(generation)
        self.plot_surface()

    def print_generation_info(self, generation):
        best_individual = min(self.population, key=lambda x: x[2])
        print(f"Geração: {generation}")
        print(f"Melhor Indivíduo: \nx = {best_individual[0]}, \ny = {best_individual[1]}, \nfitness = {best_individual[2]}")
        print('--------------------------------')

    def plot_surface(self):
        x = np.linspace(-500, 500, 100)
        y = np.linspace(-500, 500, 100)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = 837.9658 - (np.sum([x[i, j], y[i, j]] * np.sin(np.sqrt(np.abs([x[i, j], y[i, j]])))))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

        best_individual = min(self.population, key=lambda x: x[2])
        ax.scatter(best_individual[0], best_individual[1], best_individual[2], color='red', s=100, label='Melhor Indivíduo')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Cost')
        ax.set_title('Cost Function Surface')

        plt.legend()
        plt.show()

algorithm = GeneticAlgorithmProblem1()
algorithm.init_execution()
