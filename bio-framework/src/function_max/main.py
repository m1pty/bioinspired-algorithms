import math
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

import constants

def convert_to_bin(x: int) -> str:
    if x == 0:
        return '0'
    result = ''
    while x != 0:
        rect = x % 2
        result = f'{rect}' + result
        x //= 2
    return result


def generate_chromosome(x: int, l: int):
    b = convert_to_bin(x)
    return (l - len(b)) * '0' + b

def repair_chromosome(chrom: str, min_idx: int, max_idx: int, length: int) -> str:
    val = int(chrom, 2)
    if val < min_idx:
        val = min_idx
    elif val > max_idx:
        val = max_idx
    return generate_chromosome(val, length)

def initial_setup(n: int, length: int, l_border: int, r_border: int) -> list:
    samples = random.sample(
        range(
            int(l_border / constants.accuracy), 
            int(r_border / constants.accuracy) + 1
        ), 
        k=n
    )
    return list(generate_chromosome(x, length) for x in samples)

def calc_real_fitness(chromosomes: list, func) -> list:
    pre_norm = []
    for chromosome in chromosomes:
        x = int(chromosome, base=2) * constants.accuracy
        try:
            pre_norm.append(func(x))
        except ValueError:
            pre_norm.append(-1e10)
    return pre_norm

def calc_fitness(chromosomes: list, func) -> list:
    pre_norm = []
    for chromosome in chromosomes:
        x = int(chromosome, base=2) * constants.accuracy
        try:
            pre_norm.append(func(x))
        except ValueError:
            pre_norm.append(-1e10)
    min_fitness = min(pre_norm)
    return [chromo_fitness - min_fitness + 1e-6 for chromo_fitness in pre_norm]


def reproduce_crossover(samples: list, fitness: list) -> list:
    assert len(samples) == len(fitness)
    pop_size = len(samples)
    reproduce_result = []
    chromo_len = len(samples[0]) if samples else 0
    min_idx = int(constants.borders[0] / constants.accuracy)
    max_idx = int(constants.borders[1] / constants.accuracy)
    for _ in range(pop_size // 2):
        parents = random.choices(samples, weights=fitness, k=2)
        p1, p2 = parents[0], parents[1]
        if random.random() < constants.p_crossingover:
            crossover_point = random.randint(1, chromo_len - 1)
            c1 = p1[:crossover_point] + p2[crossover_point:]
            c2 = p2[:crossover_point] + p1[crossover_point:]
            c1 = repair_chromosome(c1, min_idx, max_idx, chromo_len)
            c2 = repair_chromosome(c2, min_idx, max_idx, chromo_len)
            reproduce_result.extend([c1, c2])
        else:
            reproduce_result.extend([p1, p2])
    if pop_size % 2 == 1:
        p = random.choices(samples, weights=fitness, k=1)[0]
        reproduce_result.append(p)
    return reproduce_result


def mutate(samples: list) -> list:
    chromo_length = len(samples[0]) if samples else 0
    mutated = []
    min_idx = int(constants.borders[0] / constants.accuracy)
    max_idx = int(constants.borders[1] / constants.accuracy)
    for index in range(len(samples)):
        is_mutating_random = random.choices(
            [True, False], weights=[constants.p_mutation_random, 1-constants.p_mutation_random], k=1
        )[0]
        is_mutating_swap = random.choices(
            [True, False], weights=[constants.p_mutation_swap, 1-constants.p_mutation_swap], k=1
        )[0]
        is_mutating_reverse = random.choices(
           [True, False], weights=[constants.p_mutation_reverse, 1-constants.p_mutation_reverse], k=1 
        )[0]

        chromosome = samples[index]
        if is_mutating_random:
            mutation_index = random.choice(range(0, chromo_length))
            chromosome = chromosome[:mutation_index] + \
                str(1-int(chromosome[mutation_index])) + \
                chromosome[mutation_index+1:]
        
        if is_mutating_swap:
            indexes = random.sample(range(0, chromo_length), k=2)
            i, j = min(indexes), max(indexes)
            chromosome = chromosome[:i] + chromosome[j] + chromosome[i+1:j] + chromosome[i] + chromosome[j+1:]

        if is_mutating_reverse:
            indexes = random.sample(range(0, chromo_length), k=2)
            i, j = min(indexes), max(indexes)
            chromosome = chromosome[:i] + chromosome[i:j][::-1] + chromosome[j:]

        chromosome = repair_chromosome(chromosome, min_idx, max_idx, chromo_length)
        mutated.append(chromosome)
    return mutated


def lifecycle(samples: list):
    fitness_list = calc_fitness(samples, constants.function)
    crossed = reproduce_crossover(samples, fitness_list)
    mutated = mutate(crossed)
    return mutated


def gen_plot(samples: list, real_fitness: list, n_gen: int):
    plt.figure(figsize=(19.2, 5.4))
    x_vals = np.linspace(constants.borders[0], constants.borders[1], 100000)
    y_vals = [constants.function(x) for x in x_vals]
    plt.plot(x_vals, y_vals, label='График функции f(x)=ln(x)*cos(3x-15)')
            
    chrom_x = [int(chrom, 2) * constants.accuracy for chrom in samples]
    chrom_y = real_fitness
    plt.scatter(chrom_x, chrom_y, color='red', label='Популяция')
    plt.vlines(chrom_x, ymin=0, ymax=chrom_y, color='red', linestyle='--', alpha=0.5)
            
    plt.legend()
    plt.title(f'Поколение {n_gen}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
            
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
            
    plt.show()


def main():
    max_chromo = convert_to_bin(int(constants.borders[1] / constants.accuracy) + 1)
    chromo_len = len(max_chromo)

    samples = initial_setup(
        constants.n_chromosomes, 
        chromo_len,
        constants.borders[0],
        constants.borders[1]
    )


    best_fitness = -math.inf
    best_fitness_x = -math.inf
    for n_gen in range(constants.generations + 1):
        max_fit = -math.inf
        samples = lifecycle(samples)
        real_fitness = calc_real_fitness(samples, constants.function)
        for chromosome in samples:
            x = int(chromosome, base=2) * constants.accuracy
            cur_fit = constants.function(x)
            if (cur_fit > max_fit):
                max_fit = cur_fit
            
            if (cur_fit > best_fitness):
                best_fitness = cur_fit
                best_fitness_x = x

        max_fit = max(real_fitness)
        # print(f"[>] gen {n_gen}: ({best_fitness_x}, {best_fitness})")


        # if n_gen % constants.show_graph == 0:
        #     gen_plot(samples, real_fitness, n_gen)

    
    print(f"({best_fitness_x}, {best_fitness})")

main()