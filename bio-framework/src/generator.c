#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void shuffle(int *array, int n) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5 && argc != 6) {
        printf("[!] USAGE: ./generator <max_len> <weight_limit> <number_of> <n_tasks> [seed]\n");
        printf("    <max_len>      - vector length\n");
        printf("    <weight_limit> - max item weight (weights are generated in [1, weight_limit])\n");
        printf("    <number_of>    - number of vectors\n");
        printf("    <n_tasks>      - number of tasks per vector\n");
        printf("    [seed]         - optional random seed\n");
        return 0;
    }

    int max_len = atoi(argv[1]);
    int weight_limit = atoi(argv[2]);
    int number_of = atoi(argv[3]);
    int n_tasks = atoi(argv[4]);
    unsigned int seed = (argc == 6) ? (unsigned int)strtoul(argv[5], NULL, 10)
                                    : (unsigned int)time(NULL);

    if (max_len <= 0 || weight_limit <= 0 || number_of <= 0 || n_tasks <= 0) {
        fprintf(stderr, "[!] all numeric arguments must be positive\n");
        return 1;
    }

    srand(seed);

    unsigned int **examples = (unsigned int **)calloc((size_t)number_of, sizeof(unsigned int *));
    int *indices = (int *)malloc((size_t)max_len * sizeof(int));
    if (!examples || !indices) {
        fprintf(stderr, "[!] memory allocation failed\n");
        free(examples);
        free(indices);
        return 1;
    }

    for (int j = 0; j < max_len; ++j) {
        indices[j] = j;
    }

    for (int i = 0; i < number_of; ++i) {
        examples[i] = (unsigned int *)calloc((size_t)max_len, sizeof(unsigned int));
        if (!examples[i]) {
            fprintf(stderr, "[!] memory allocation failed\n");
            for (int k = 0; k < i; ++k) free(examples[k]);
            free(examples);
            free(indices);
            return 1;
        }

        for (int j = 0; j < max_len; ++j) {
            examples[i][j] = (unsigned int)(rand() % weight_limit + 1);
        }
    }

    const double p_values[5] = {0.1, 0.2, 0.3, 0.4, 0.5};

    for (int i = 0; i < number_of; ++i) {
        for (int j = 0; j < max_len; ++j) {
            if (j) printf(" ");
            printf("%u", examples[i][j]);
        }
        printf("\n");

        for (int t = 0; t < n_tasks; ++t) {
            double p = p_values[t % 5];
            int k = (int)lround(p * max_len);
            if (k < 1) k = 1;
            if (k > max_len) k = max_len;

            shuffle(indices, max_len);

            unsigned long long sum = 0;
            for (int j = 0; j < k; ++j) {
                sum += examples[i][indices[j]];
            }

            printf("%.1f %llu\n", p, sum);
        }

        printf("\n");
    }

    for (int i = 0; i < number_of; ++i) {
        free(examples[i]);
    }
    free(examples);
    free(indices);

    return 0;
}