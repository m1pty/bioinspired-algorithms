#ifndef GA_H
#define GA_H

#include <stdint.h>

#define VARIANT_N 20
#define VARIANT_AMAX 1048576u
#define VARIANT_MODULUS 1048577ull

#define MAX_N 24
#define MAX_LINE 4096
#define MAX_VECTORS 2048
#define MAX_TASKS_TOTAL 65536

typedef struct {
    int id;
    int n;
    uint32_t weights[MAX_N];
} Vector;

typedef struct {
    int id;
    int vector_id;
    int n;
    uint64_t target;
    double p;
    int task_index_inside_vector;
} Problem;

typedef struct {
    int problem_id;
    int vector_id;
    double time_first_ms;
    double time_all_ms;
    int num_solutions;
    uint32_t first_solution_mask;
} BFResult;

typedef struct {
    int problem_id;
    int vector_id;
    double time_ms;
    double best_fitness;
    uint64_t min_error;
    int last_generation;
    char stop_reason[32];
    uint32_t best_mask;
    int exact_found;
} GAResult;

typedef struct {
    uint64_t state;
} RNG;

typedef double   (*fitness_fn)(uint64_t err);
typedef int      (*selection_fn)(const double *fits, int pop_size, int tournament_k, RNG *rng);
typedef uint32_t (*crossover_fn)(uint32_t p1, uint32_t p2, int n, double pc, RNG *rng);
typedef uint32_t (*mutation_fn)(uint32_t child, int n, double pm, RNG *rng);

typedef struct {
    const char *name;
    selection_fn fn;
} SelectionEntry;

typedef struct {
    const char *name;
    crossover_fn fn;
} CrossoverEntry;

typedef struct {
    const char *name;
    mutation_fn fn;
} MutationEntry;

typedef struct {
    const char *name;
    fitness_fn fn;
} FitnessEntry;

typedef struct {
    int pop_size;
    double pc;
    double pm;
    int tournament_k;
    int stagnation_limit;
    uint64_t modulus;
    int use_average_stop;
} GAConfig;

typedef struct {
    int combo_id;
    const char *fitness_name;
    const char *selection_name;
    const char *crossover_name;
    const char *mutation_name;

    double avg_bf_first_ms;
    double avg_bf_all_ms;
    double avg_ga_ms;
    double avg_error;
    double avg_generation;
    double exact_ratio;
    double score;
} ComboSummary;

/* RNG */
void rng_seed(RNG *rng, uint64_t seed);
uint64_t rng_next_u64(RNG *rng);
uint32_t rng_next_u32(RNG *rng);
double rng_next_double(RNG *rng);
int rng_next_int(RNG *rng, int bound);

/* registries */
const SelectionEntry *get_selection_registry(int *count);
const CrossoverEntry *get_crossover_registry(int *count);
const MutationEntry *get_mutation_registry(int *count);
const FitnessEntry *get_fitness_registry(int *count);

/* utils */
uint64_t error_mod(uint64_t sum, uint64_t target, uint64_t modulus);
int is_solution_mod(uint64_t sum, uint64_t target, uint64_t modulus);
uint64_t sum_mask(const Vector *v, uint32_t mask);
void mask_to_binary(uint32_t mask, int n, char *out);

#endif