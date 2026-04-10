from math import log, e, cos
function = lambda x: log(x, e) * cos(3 * x - 15)
p_crossingover = 0.9
accuracy = 0.00001
borders = (1, 10)

n_chromosomes = 100
generations = 500

p_mutation_random   = 0.1
p_mutation_swap     = 0.05
p_mutation_reverse  = 0.01

show_graph = 100