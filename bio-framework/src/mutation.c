#include "../include/ga.h"

static uint32_t mutation_bitflip(uint32_t child, int n, double pm, RNG *rng) {
    for (int i = 0; i < n; ++i) {
        if (rng_next_double(rng) < pm) {
            child ^= (1u << i);
        }
    }
    return child;
}

static uint32_t mutation_onebit(uint32_t child, int n, double pm, RNG *rng) {
    if (rng_next_double(rng) < pm) {
        int bit = rng_next_int(rng, n);
        child ^= (1u << bit);
    }
    return child;
}

static uint32_t mutation_twobit(uint32_t child, int n, double pm, RNG *rng) {
    if (rng_next_double(rng) < pm) {
        int b1 = rng_next_int(rng, n);
        int b2 = rng_next_int(rng, n);
        child ^= (1u << b1);
        child ^= (1u << b2);
    }
    return child;
}

static const MutationEntry REGISTRY[] = {
    {"bitflip", mutation_bitflip},
    {"onebit",  mutation_onebit},
    {"twobit",  mutation_twobit}
};

const MutationEntry *get_mutation_registry(int *count) {
    *count = (int)(sizeof(REGISTRY) / sizeof(REGISTRY[0]));
    return REGISTRY;
}