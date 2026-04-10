#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir_if_needed(path) _mkdir(path)
#else
#define mkdir_if_needed(path) mkdir(path, 0755)
#endif

#include "../include/ga.h"

// RNG seed
void rng_seed(RNG *rng, uint64_t seed) {
    if (seed == 0) seed = 0x9E3779B97F4A7C15ULL;
    rng->state = seed;
}

uint64_t rng_next_u64(RNG *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 2685821657736338717ULL;
}

uint32_t rng_next_u32(RNG *rng) {
    return (uint32_t)(rng_next_u64(rng) >> 32);
}

double rng_next_double(RNG *rng) {
    return (rng_next_u64(rng) >> 11) * (1.0 / 9007199254740992.0);
}

int rng_next_int(RNG *rng, int bound) {
    return (int)(rng_next_u32(rng) % (uint32_t)bound);
}

// =============================== HELPERS ===============================

static int ensure_dir(const char *path) {
    int rc = mkdir_if_needed(path);
    if (rc == 0) return 1;
    return 1;
}

uint64_t error_mod(uint64_t sum, uint64_t target, uint64_t modulus) {
    uint64_t sm = sum % modulus;
    uint64_t tm = target % modulus;
    uint64_t diff = (sm >= tm) ? (sm - tm) : (tm - sm);
    uint64_t comp = modulus - diff;
    return (comp < diff) ? comp : diff;
}

int is_solution_mod(uint64_t sum, uint64_t target, uint64_t modulus) {
    return (sum % modulus) == (target % modulus);
}

uint64_t sum_mask(const Vector *v, uint32_t mask) {
    uint64_t s = 0;
    for (int i = 0; i < v->n; ++i) {
        if (mask & (1u << i)) s += v->weights[i];
    }
    return s;
}

void mask_to_binary(uint32_t mask, int n, char *out) {
    for (int i = 0; i < n; ++i) {
        out[n - 1 - i] = (mask & (1u << i)) ? '1' : '0';
    }
    out[n] = '\0';
}

static void sort_vector(uint32_t *a, int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (a[i] > a[j]) {
                uint32_t t = a[i];
                a[i] = a[j];
                a[j] = t;
            }
        }
    }
}

static double average_fit(const double *fits, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += fits[i];
    return s / (double)n;
}

// =============================== FITNESS COUNTERS ===============================

static double fitness_ring(uint64_t err) {
    return (1.0 - 2 * (double)err / VARIANT_MODULUS);
}

static double fitness_linear(uint64_t err) {
    return 1.0 / (1.0 + (double)err);
}

static double fitness_logarithmic(uint64_t err) {
    return 1.0 / (1.0 + log2(1.0 + (double)err));
}

static const FitnessEntry FITNESS_REGISTRY[] = {
    {"linear",  fitness_linear},
    {"log",     fitness_logarithmic},
    {"ring",    fitness_ring}
};

const FitnessEntry *get_fitness_registry(int *count) {
    *count = (int)(sizeof(FITNESS_REGISTRY) / sizeof(FITNESS_REGISTRY[0]));
    return FITNESS_REGISTRY;
}

// =============================== PARSER ===============================

static int parse_vector_line(const char *line, Vector *v) {
    char buf[MAX_LINE];
    strncpy(buf, line, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    int n = 0;
    char *tok = strtok(buf, " \t\r\n");
    while (tok && n < MAX_N) {
        v->weights[n++] = (uint32_t)strtoul(tok, NULL, 10);
        tok = strtok(NULL, " \t\r\n");
    }
    v->n = n;
    return (n > 0);
}

static int load_generated_data(
    const char *filename,
    int n_tasks_per_vector,
    Vector *vectors,
    int *vector_count,
    Problem *problems,
    int *problem_count
) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("fopen");
        return 0;
    }

    char line[MAX_LINE];
    int vid = 0;
    int pid = 0;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\r') continue;

        Vector v;
        if (!parse_vector_line(line, &v)) continue;

        if (v.n != VARIANT_N) {
            fprintf(stderr, "[!] ERROR: expected n=%d, got n=%d\n", VARIANT_N, v.n);
            fclose(f);
            return 0;
        }

        if (vid >= MAX_VECTORS) {
            fprintf(stderr, "[!] ERROR: TOO MANY VECTORS\n");
            fclose(f);
            return 0;
        }

        vectors[vid].id = vid + 1;
        vectors[vid].n = v.n;
        for (int i = 0; i < v.n; ++i) vectors[vid].weights[i] = v.weights[i];

        sort_vector(vectors[vid].weights, vectors[vid].n);

        for (int t = 0; t < n_tasks_per_vector; ++t) {
            if (!fgets(line, sizeof(line), f)) {
                fprintf(stderr, "[!] ERROR: Unexpected EOF while reading tasks...\n");
                fclose(f);
                return 0;
            }

            if (line[0] == '\n' || line[0] == '\r') {
                t--;
                continue;
            }

            if (pid >= MAX_TASKS_TOTAL) {
                fprintf(stderr, "[!] ERROR: TOO MANY PROBLEMS\n");
                fclose(f);
                return 0;
            }

            double p = 0.0;
            unsigned long long target = 0ull;
            if (sscanf(line, "%lf %llu", &p, &target) != 2) {
                fprintf(stderr, "[!] ERROR: Bad task line: %s\n", line);
                fclose(f);
                return 0;
            }

            problems[pid].id = pid + 1;
            problems[pid].vector_id = vid + 1;
            problems[pid].n = VARIANT_N;
            problems[pid].target = (uint64_t)target;
            problems[pid].p = p;
            problems[pid].task_index_inside_vector = t;
            pid++;
        }

        vid++;
    }

    fclose(f);
    *vector_count = vid;
    *problem_count = pid;
    return 1;
}

// =============================== BRUTEFORCE ===============================

static BFResult solve_bruteforce(const Vector *v, const Problem *p, uint64_t modulus) {
    BFResult r;
    r.problem_id = p->id;
    r.vector_id = p->vector_id;
    r.time_first_ms = 0.0;
    r.time_all_ms = 0.0;
    r.num_solutions = 0;
    r.first_solution_mask = 0;

    uint32_t total = 1u << v->n;
    double start = omp_get_wtime();
    int found_first = 0;

    for (uint32_t mask = 0; mask < total; ++mask) {
        uint64_t sum = sum_mask(v, mask);
        if (is_solution_mod(sum, p->target, modulus)) {
            r.num_solutions++;
            if (!found_first) {
                r.time_first_ms = (omp_get_wtime() - start) * 1000.0;
                r.first_solution_mask = mask;
                found_first = 1;
            }
        }
    }

    r.time_all_ms = (omp_get_wtime() - start) * 1000.0;
    return r;
}

// =============================== GENALGO ===============================

static GAResult solve_ga(
    const Vector *v,
    const Problem *p,
    const GAConfig *cfg,
    fitness_fn fit_fn,
    selection_fn sel_fn,
    crossover_fn cx_fn,
    mutation_fn mut_fn,
    uint64_t seed,
    double max_time_ms
) {
    GAResult out;
    out.problem_id = p->id;
    out.vector_id = p->vector_id;
    out.time_ms = 0.0;
    out.best_fitness = 0.0;
    out.min_error = UINT64_MAX;
    out.last_generation = 0;
    strcpy(out.stop_reason, "unknown");
    out.best_mask = 0;
    out.exact_found = 0;

    RNG rng;
    rng_seed(&rng, seed);

    uint32_t *pop  = (uint32_t *)malloc((size_t)cfg->pop_size * sizeof(uint32_t));
    uint32_t *next = (uint32_t *)malloc((size_t)cfg->pop_size * sizeof(uint32_t));
    uint64_t *errs = (uint64_t *)malloc((size_t)cfg->pop_size * sizeof(uint64_t));
    double   *fits = (double   *)malloc((size_t)cfg->pop_size * sizeof(double));

    if (!pop || !next || !errs || !fits) {
        fprintf(stderr, "[!] GA memory allocation failed\n");
        exit(1);
    }

    uint32_t mask_limit = (1u << v->n) - 1u;

    for (int i = 0; i < cfg->pop_size; ++i) {
        pop[i] = rng_next_u32(&rng) & mask_limit;
    }

    double start = omp_get_wtime();
    double prev_metric = -1.0;
    int no_improve = 0;

    for (int gen = 0;; ++gen) {
        int best_idx = 0;
        double best_fit_gen = -1.0;

        for (int i = 0; i < cfg->pop_size; ++i) {
            uint64_t sum = sum_mask(v, pop[i]);
            errs[i] = error_mod(sum, p->target, cfg->modulus);
            fits[i] = fit_fn(errs[i]);

            if (fits[i] > best_fit_gen) {
                best_fit_gen = fits[i];
                best_idx = i;
            }
        }

        if (errs[best_idx] < out.min_error) {
            out.min_error = errs[best_idx];
            out.best_fitness = fits[best_idx];
            out.best_mask = pop[best_idx];
        }

        out.last_generation = gen;
        out.time_ms = (omp_get_wtime() - start) * 1000.0;

        if (out.min_error == 0) {
            strcpy(out.stop_reason, "exact");
            out.exact_found = 1;
            break;
        }

        double metric = cfg->use_average_stop ? average_fit(fits, cfg->pop_size) : best_fit_gen;
        if (prev_metric >= 0.0) {
            if (metric > prev_metric) no_improve = 0;
            else no_improve++;
        }
        prev_metric = metric;

        if (no_improve >= cfg->stagnation_limit) {
            strcpy(out.stop_reason, "stagnation");
            break;
        }

        if (out.time_ms >= max_time_ms) {
            strcpy(out.stop_reason, "timeout");
            break;
        }

        next[0] = pop[best_idx];

        for (int i = 1; i < cfg->pop_size; ++i) {
            int a = sel_fn(fits, cfg->pop_size, cfg->tournament_k, &rng);
            int b = sel_fn(fits, cfg->pop_size, cfg->tournament_k, &rng);
            uint32_t child = cx_fn(pop[a], pop[b], v->n, cfg->pc, &rng);
            child = mut_fn(child, v->n, cfg->pm, &rng);
            child &= mask_limit;
            next[i] = child;
        }

        uint32_t *tmp = pop;
        pop = next;
        next = tmp;
    }

    free(pop);
    free(next);
    free(errs);
    free(fits);
    return out;
}

// =============================== CSV WRITERS ===============================

static void write_vectors_csv(const char *out_dir, const Vector *vectors, int vector_count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/vectors.csv", out_dir);

    FILE *f = fopen(path, "w");
    if (!f) { perror("vectors.csv"); exit(1); }

    fprintf(f, "vector_id,n");
    for (int i = 0; i < VARIANT_N; ++i) fprintf(f, ",w%d", i + 1);
    fprintf(f, "\n");

    for (int i = 0; i < vector_count; ++i) {
        fprintf(f, "%d,%d", vectors[i].id, vectors[i].n);
        for (int j = 0; j < vectors[i].n; ++j) fprintf(f, ",%u", vectors[i].weights[j]);
        fprintf(f, "\n");
    }

    fclose(f);
}


static void write_problems_csv(const char *out_dir, const Problem *problems, int problem_count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/problems.csv", out_dir);

    FILE *f = fopen(path, "w");
    if (!f) { perror("problems.csv"); exit(1); }

    fprintf(f, "problem_id,vector_id,n,target,p,task_index_inside_vector\n");
    for (int i = 0; i < problem_count; ++i) {
        fprintf(f, "%d,%d,%d,%llu,%.1f,%d\n",
                problems[i].id,
                problems[i].vector_id,
                problems[i].n,
                (unsigned long long)problems[i].target,
                problems[i].p,
                problems[i].task_index_inside_vector);
    }

    fclose(f);
}


static void write_bruteforce_csv(const char *out_dir, const BFResult *results, int count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/bruteforce_results.csv", out_dir);

    FILE *f = fopen(path, "w");
    if (!f) { perror("bruteforce_results.csv"); exit(1); }

    fprintf(f, "problem_id,vector_id,time_first_ms,time_all_ms,num_solutions,first_solution_mask_bin\n");
    for (int i = 0; i < count; ++i) {
        char maskbuf[VARIANT_N + 1];
        mask_to_binary(results[i].first_solution_mask, VARIANT_N, maskbuf);
        fprintf(f, "%d,%d,%.6f,%.6f,%d,%s\n",
                results[i].problem_id,
                results[i].vector_id,
                results[i].time_first_ms,
                results[i].time_all_ms,
                results[i].num_solutions,
                maskbuf);
    }

    fclose(f);
}


static void write_ga_results_csv_header(FILE *f) {
    fprintf(f,
        "combo_id,problem_id,vector_id,p,target,"
        "fitness_type,selection_type,crossover_type,mutation_type,"
        "pop_size,pc,pm,tournament_k,stagnation_limit,modulus,seed,"
        "bf_time_all_ms,ga_time_ms,best_fitness,min_error,last_generation,stop_reason,exact_found,best_mask_bin\n");
}


static void append_ga_results_csv(
    FILE *f,
    const GAResult *ga_results,
    const BFResult *bf_results,
    const Problem *problems,
    int problem_count,
    const GAConfig *cfg,
    int combo_id,
    const char *fitness_name,
    const char *selection_name,
    const char *crossover_name,
    const char *mutation_name,
    uint64_t master_seed
) {
    for (int i = 0; i < problem_count; ++i) {
        char maskbuf[VARIANT_N + 1];
        mask_to_binary(ga_results[i].best_mask, VARIANT_N, maskbuf);
        uint64_t seed = master_seed + (uint64_t)(combo_id + 1) * 10000019ULL + (uint64_t)(i + 1) * 1000003ULL;

        fprintf(f,
            "%d,%d,%d,%.1f,%llu,"
            "%s,%s,%s,%s,"
            "%d,%.6f,%.6f,%d,%d,%llu,%llu,"
            "%.6f,%.6f,%.12f,%llu,%d,%s,%d,%s\n",
            combo_id,
            ga_results[i].problem_id,
            ga_results[i].vector_id,
            problems[i].p,
            (unsigned long long)problems[i].target,
            fitness_name,
            selection_name,
            crossover_name,
            mutation_name,
            cfg->pop_size,
            cfg->pc,
            cfg->pm,
            cfg->tournament_k,
            cfg->stagnation_limit,
            (unsigned long long)cfg->modulus,
            (unsigned long long)seed,
            bf_results[i].time_all_ms,
            ga_results[i].time_ms,
            ga_results[i].best_fitness,
            (unsigned long long)ga_results[i].min_error,
            ga_results[i].last_generation,
            ga_results[i].stop_reason,
            ga_results[i].exact_found,
            maskbuf
        );
    }
}


static void write_combo_summary_csv(const char *out_dir, const ComboSummary *items, int count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/combo_summary.csv", out_dir);

    FILE *f = fopen(path, "w");
    if (!f) { perror("combo_summary.csv"); exit(1); }

    fprintf(f, "combo_id,fitness_type,selection_type,crossover_type,mutation_type,avg_bf_first_ms,avg_bf_all_ms,avg_ga_ms,avg_error,avg_generation,exact_ratio,score\n");
    for (int i = 0; i < count; ++i) {
        fprintf(f, "%d,%s,%s,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                items[i].combo_id,
                items[i].fitness_name,
                items[i].selection_name,
                items[i].crossover_name,
                items[i].mutation_name,
                items[i].avg_bf_first_ms,
                items[i].avg_bf_all_ms,
                items[i].avg_ga_ms,
                items[i].avg_error,
                items[i].avg_generation,
                items[i].exact_ratio,
                items[i].score);
    }

    fclose(f);
}

static void write_run_info_csv(
    const char *out_dir,
    const char *input_file,
    int n_tasks_per_vector,
    int vector_count,
    int problem_count,
    const GAConfig *cfg,
    uint64_t seed,
    int threads
) {
    char path[512];
    snprintf(path, sizeof(path), "%s/run_info.csv", out_dir);

    FILE *f = fopen(path, "w");
    if (!f) { perror("run_info.csv"); exit(1); }

    fprintf(f, "key,value\n");
    fprintf(f, "variant,7\n");
    fprintf(f, "input_file,%s\n", input_file);
    fprintf(f, "vector_count,%d\n", vector_count);
    fprintf(f, "problem_count,%d\n", problem_count);
    fprintf(f, "n,%d\n", VARIANT_N);
    fprintf(f, "amax,%u\n", VARIANT_AMAX);
    fprintf(f, "modulus,%llu\n", (unsigned long long)VARIANT_MODULUS);
    fprintf(f, "n_tasks_per_vector,%d\n", n_tasks_per_vector);
    fprintf(f, "pop_size,%d\n", cfg->pop_size);
    fprintf(f, "pc,%.6f\n", cfg->pc);
    fprintf(f, "pm,%.6f\n", cfg->pm);
    fprintf(f, "tournament_k,%d\n", cfg->tournament_k);
    fprintf(f, "stagnation_limit,%d\n", cfg->stagnation_limit);
    fprintf(f, "use_average_stop,%d\n", cfg->use_average_stop);
    fprintf(f, "seed,%llu\n", (unsigned long long)seed);
    fprintf(f, "omp_threads,%d\n", threads);

    fclose(f);
}

// =============================== USAGE HELPER ===============================

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s <generated_file> <n_tasks_per_vector> <out_dir> [seed] [threads]\n", prog);
    printf("\n");
    printf("Example:\n");
    printf("  %s generated.txt 10 logs 12345 8\n", prog);
}

// =============================== MAIN ===============================

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        usage(argv[0]);
        return 1;
    }

    const char *generated_file = argv[1];
    int n_tasks_per_vector = atoi(argv[2]);
    const char *out_dir = argv[3];
    uint64_t master_seed = (argc >= 5) ? strtoull(argv[4], NULL, 10) : 12345ULL;
    int threads = (argc >= 6) ? atoi(argv[5]) : omp_get_max_threads();

    if (threads < 1) threads = 1;
    omp_set_num_threads(threads);

    ensure_dir(out_dir);

    Vector vectors[MAX_VECTORS];
    Problem *problems = (Problem *)calloc(MAX_TASKS_TOTAL, sizeof(Problem));
    BFResult *bf_results = (BFResult *)calloc(MAX_TASKS_TOTAL, sizeof(BFResult));

    if (!problems || !bf_results) {
        fprintf(stderr, "[!] Memory allocation failed\n");
        free(problems);
        free(bf_results);
        return 1;
    }

    int vector_count = 0;
    int problem_count = 0;

    if (!load_generated_data(generated_file, n_tasks_per_vector, vectors, &vector_count, problems, &problem_count)) {
        free(problems);
        free(bf_results);
        return 1;
    }


    GAConfig cfg;
    cfg.pop_size = 1000;
    cfg.pc = 0.5;
    cfg.pm = 1.0 / 24.0;
    cfg.tournament_k = 3;
    cfg.stagnation_limit = 2;
    cfg.modulus = VARIANT_MODULUS;
    cfg.use_average_stop = 1;

    printf("n=%d amax=%u modulus=%llu\n", VARIANT_N, VARIANT_AMAX, (unsigned long long)VARIANT_MODULUS);
    printf("Loaded vectors=%d problems=%d threads=%d\n", vector_count, problem_count, threads);

    // bruteforce for once and then use
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < problem_count; ++i) {
        const Vector *v = &vectors[problems[i].vector_id - 1];
        bf_results[i] = solve_bruteforce(v, &problems[i], cfg.modulus);
    }

    write_vectors_csv(out_dir, vectors, vector_count);
    write_problems_csv(out_dir, problems, problem_count);
    write_bruteforce_csv(out_dir, bf_results, problem_count);

    char ga_path[512];
    snprintf(ga_path, sizeof(ga_path), "%s/ga_results.csv", out_dir);
    FILE *ga_log = fopen(ga_path, "w");
    if (!ga_log) {
        perror("ga_results.csv");
        free(problems);
        free(bf_results);
        return 1;
    }
    write_ga_results_csv_header(ga_log);

    int fit_count, sel_count, cx_count, mut_count;
    const FitnessEntry   *fits = get_fitness_registry(&fit_count);
    const SelectionEntry *sels = get_selection_registry(&sel_count);
    const CrossoverEntry *cxs  = get_crossover_registry(&cx_count);
    const MutationEntry  *muts = get_mutation_registry(&mut_count);

    int combo_count = fit_count * sel_count * cx_count * mut_count;
    ComboSummary *summary = (ComboSummary *)calloc((size_t)combo_count, sizeof(ComboSummary));
    if (!summary) {
        fprintf(stderr, "Summary allocation failed\n");
        fclose(ga_log);
        free(problems);
        free(bf_results);
        return 1;
    }

    int combo_id = 0;
    double best_score = 1e300;
    int best_combo = -1;

    for (int fi = 0; fi < fit_count; ++fi) {
        for (int si = 0; si < sel_count; ++si) {
            for (int ci = 0; ci < cx_count; ++ci) {
                for (int mi = 0; mi < mut_count; ++mi) {
                    printf("Combo %d/%d: fit=%s sel=%s cx=%s mut=%s\n",
                           combo_id + 1, combo_count,
                           fits[fi].name, sels[si].name, cxs[ci].name, muts[mi].name);

                    GAResult *ga_results = (GAResult *)calloc((size_t)problem_count, sizeof(GAResult));
                    if (!ga_results) {
                        fprintf(stderr, "GA results allocation failed\n");
                        fclose(ga_log);
                        free(summary);
                        free(problems);
                        free(bf_results);
                        return 1;
                    }

                    #pragma omp parallel for schedule(dynamic)
                    for (int i = 0; i < problem_count; ++i) {
                        const Vector *v = &vectors[problems[i].vector_id - 1];
                        double max_time_ms = 2.0 * bf_results[i].time_all_ms;
                        if (max_time_ms < 1.0) max_time_ms = 1.0;

                        uint64_t seed = master_seed
                                      + (uint64_t)(combo_id + 1) * 10000019ULL
                                      + (uint64_t)(i + 1) * 1000003ULL;

                        ga_results[i] = solve_ga(
                            v,
                            &problems[i],
                            &cfg,
                            fits[fi].fn,
                            sels[si].fn,
                            cxs[ci].fn,
                            muts[mi].fn,
                            seed,
                            max_time_ms
                        );
                    }

                    double sum_bf_first = 0.0;
                    double sum_bf_all = 0.0;
                    double sum_ga_time = 0.0;
                    double sum_err = 0.0;
                    double sum_gen = 0.0;
                    int exact_count = 0;

                    for (int i = 0; i < problem_count; ++i) {
                        sum_bf_first += bf_results[i].time_first_ms;
                        sum_bf_all   += bf_results[i].time_all_ms;
                        sum_ga_time  += ga_results[i].time_ms;
                        sum_err      += (double)ga_results[i].min_error;
                        sum_gen      += (double)ga_results[i].last_generation;
                        exact_count  += ga_results[i].exact_found;
                    }

                    summary[combo_id].combo_id = combo_id;
                    summary[combo_id].fitness_name = fits[fi].name;
                    summary[combo_id].selection_name = sels[si].name;
                    summary[combo_id].crossover_name = cxs[ci].name;
                    summary[combo_id].mutation_name = muts[mi].name;
                    summary[combo_id].avg_bf_first_ms = sum_bf_first / problem_count;
                    summary[combo_id].avg_bf_all_ms   = sum_bf_all / problem_count;
                    summary[combo_id].avg_ga_ms       = sum_ga_time / problem_count;
                    summary[combo_id].avg_error       = sum_err / problem_count;
                    summary[combo_id].avg_generation  = sum_gen / problem_count;
                    summary[combo_id].exact_ratio     = (double)exact_count / (double)problem_count;

                    summary[combo_id].score =
                        summary[combo_id].avg_error
                        + 0.01 * summary[combo_id].avg_ga_ms
                        + 1000000.0 * (1.0 - summary[combo_id].exact_ratio);

                    if (summary[combo_id].score < best_score) {
                        best_score = summary[combo_id].score;
                        best_combo = combo_id;
                    }

                    append_ga_results_csv(
                        ga_log,
                        ga_results,
                        bf_results,
                        problems,
                        problem_count,
                        &cfg,
                        combo_id,
                        fits[fi].name,
                        sels[si].name,
                        cxs[ci].name,
                        muts[mi].name,
                        master_seed
                    );

                    free(ga_results);
                    combo_id++;
                }
            }
        }
    }

    fclose(ga_log);
    write_combo_summary_csv(out_dir, summary, combo_count);
    write_run_info_csv(out_dir, generated_file, n_tasks_per_vector, vector_count, problem_count, &cfg, master_seed, threads);

    if (best_combo >= 0) {
        printf("\nBest combo:\n");
        printf("  combo_id=%d\n", summary[best_combo].combo_id);
        printf("  fitness=%s\n", summary[best_combo].fitness_name);
        printf("  selection=%s\n", summary[best_combo].selection_name);
        printf("  crossover=%s\n", summary[best_combo].crossover_name);
        printf("  mutation=%s\n", summary[best_combo].mutation_name);
        printf("  avg_ga_ms=%.6f\n", summary[best_combo].avg_ga_ms);
        printf("  avg_error=%.6f\n", summary[best_combo].avg_error);
        printf("  exact_ratio=%.6f\n", summary[best_combo].exact_ratio);
        printf("  score=%.6f\n", summary[best_combo].score);
    }

    printf("\n[+] DONE: logs written to %s/\n", out_dir);

    free(summary);
    free(problems);
    free(bf_results);
    return 0;
}