#include "../include/ga.h"

static int selection_tournament(const double *fits, int pop_size, int tournament_k, RNG *rng) {
    int k = tournament_k > 1 ? tournament_k : 2;
    int best = rng_next_int(rng, pop_size);

    for (int i = 1; i < k; ++i) {
        int cand = rng_next_int(rng, pop_size);
        if (fits[cand] > fits[best]) best = cand;
    }
    return best;
}

static int selection_roulette(const double *fits, int pop_size, int tournament_k, RNG *rng) {
    (void)tournament_k;

    double sum = 0.0;
    for (int i = 0; i < pop_size; ++i) sum += fits[i];

    if (sum <= 0.0) return rng_next_int(rng, pop_size);

    double r = rng_next_double(rng) * sum;
    double acc = 0.0;

    for (int i = 0; i < pop_size; ++i) {
        acc += fits[i];
        if (acc >= r) return i;
    }
    return pop_size - 1;
}

static const SelectionEntry REGISTRY[] = {
    {"tournament", selection_tournament},
    {"roulette",   selection_roulette}
};

const SelectionEntry *get_selection_registry(int *count) {
    *count = (int)(sizeof(REGISTRY) / sizeof(REGISTRY[0]));
    return REGISTRY;
}