#include "../include/ga.h"

static uint32_t crossover_uniform(uint32_t p1, uint32_t p2, int n, double pc, RNG *rng) {
    if (rng_next_double(rng) >= pc) return p1;

    uint32_t child = 0;
    for (int i = 0; i < n; ++i) {
        uint32_t bit = (rng_next_int(rng, 2) == 0) ? ((p1 >> i) & 1u) : ((p2 >> i) & 1u);
        child |= (bit << i);
    }
    return child;
}

static uint32_t crossover_onepoint(uint32_t p1, uint32_t p2, int n, double pc, RNG *rng) {
    if (rng_next_double(rng) >= pc) return p1;

    int point = 1 + rng_next_int(rng, n - 1);
    uint32_t left_mask = (1u << point) - 1u;
    return (p1 & left_mask) | (p2 & ~left_mask);
}

static uint32_t crossover_twopoint(uint32_t p1, uint32_t p2, int n, double pc, RNG *rng) {
    if (rng_next_double(rng) >= pc) return p1;

    int a = rng_next_int(rng, n);
    int b = rng_next_int(rng, n);
    if (a > b) {
        int t = a;
        a = b;
        b = t;
    }

    if (a == b) {
        if (b < n - 1) b++;
        else if (a > 0) a--;
    }

    uint32_t child = 0;
    for (int i = 0; i < n; ++i) {
        uint32_t bit = (i >= a && i < b) ? ((p2 >> i) & 1u) : ((p1 >> i) & 1u);
        child |= (bit << i);
    }
    return child;
}

static const CrossoverEntry REGISTRY[] = {
    {"uniform",  crossover_uniform},
    {"onepoint", crossover_onepoint},
    {"twopoint", crossover_twopoint}
};

const CrossoverEntry *get_crossover_registry(int *count) {
    *count = (int)(sizeof(REGISTRY) / sizeof(REGISTRY[0]));
    return REGISTRY;
}