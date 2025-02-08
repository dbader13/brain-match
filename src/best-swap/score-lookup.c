#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <omp.h>

#include "heap.h"
#include "xoshiro.h"
#include "utils.h"
#include "parallel_hungarian_subset.h"
#include "interrupts.h"

#define CURRENT_LOG_LEVEL LOG_LEVEL_INFO

const int HEAP_CAPACITY = 200 * 1000;
const bool ACCEPT_ZERO_DELTA = true;
const int SAVE_FREQUENCY = 800;
const int ITERATIONS = 2 * 1000 * 1000;
const int REINSERT_INTERVAL = 50 * 1000; //HEAP_CAPACITY / 4; only needs to be HEAP_CAPACITY / 2 (or possibly HEAP_CAPACITY) but just for testing
const int UPDATE_INTERVAL = 2000;

const bool PROBABILISTIC = true;
const bool SIMULATED_ANNEALING = true;
const bool HUNGARIAN = true;

const int PROBABILISTIC_INTERVAL = 300;
const int PROBABILISTIC_SWAPS = 5;
const double PROBABILISTIC_TEMPERATURE = 1.0;

const int SIMULATED_ANNEALING_INTERVAL = 1000;
const int SIMULATED_ANNEALING_ITERATIONS = 50 * 1000 * 1000;
const double SIMULATED_ANNEALING_TMIN = 0.05;
const double SIMULATED_ANNEALING_TMAX = 0.5;

const int HUNGARIAN_ITERS = 2000;
const int HUNGARIAN_INTERVAL = 2500;

void sigint_handler(int sig) {
    interrupted = 1;

    fprintf(stderr, "\nReceived interrupt signal. Cleaning up...\n");
}

short *load_lookup(const char *filename, SwapHeap *heap) {
    LOG_INFO("Loading score lookup...");

    FILE *file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filename);
        return NULL;
    }

    short *lookup =
        (short *)calloc((NUM_NODES + 1) * (NUM_NODES + 1), sizeof(short));

    // int maxPrints = 100;
    // int prints = 0;

    for (int i = 1; i <= NUM_NODES; i++) {
        if (i % 1000 == 0){
            print_progress(i, NUM_NODES, "Loading lookup...");
            if(interrupted != 0){
                fclose(file);
                free(lookup);
                return NULL;
            }
        } 
        for (int j = 1; j <= NUM_NODES; j++) {
            if (fscanf(file, "%hd,", &lookup[i * (NUM_NODES+1) + j]) != 1) {
                fprintf(stderr, "Error reading file: %s at row %d, column %d\n",
                        filename, i, j);
                fclose(file);
                free(lookup);
                return NULL;
            }
        }
    }

    fclose(file);
    return lookup;
}

void reinsert_into_heap(SwapHeap *heap, short *lookup, bool acceptZero) {
    for (int i = 1; i <= NUM_NODES; i++) {
        for (int j = 1; j <= NUM_NODES; j++) {
            if (i > j && (lookup[i * (NUM_NODES+1) + j] > 0 || (acceptZero && lookup[i * (NUM_NODES+1) + j] == 0))) {
                insert_swap(heap, (Swap){i, j, lookup[i * (NUM_NODES+1) + j]});
            }
        }
    }
}

void empty_heap(SwapHeap *heap) {
    Swap dummy;
    while (heap->size > 0) {
        extract_max(heap, &dummy);
    }
    LOG_INFO("Emptied heap, size: %d", heap->size);
}

void save_lookup(const char *filename, short *lookup) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        LOG_ERROR("Error creating file: %s", filename);
        fclose(file);
        return;
    }

    for (int i = 0; i < NUM_NODES; i++) {
        if(i % 1000 == 0 && interrupted != 0){
            fclose(file);
            return;
        }
        for (int j = 0; j < NUM_NODES - 1; j++) {
            fprintf(file, "%d,", lookup[i * (NUM_NODES+1) + j]);
        }
        fprintf(file, "%d", lookup[i * (NUM_NODES+1) + NUM_NODES]);
        fprintf(file, "\n");
    }

    fclose(file);
}

void update_lookup_and_heap(short *lookup, SwapHeap *swapHeap, Graph *gm,
                         Graph *gf, int *bestMapping, int nodeM1, int nodeM2) {
    // Use OpenMP to parallelize the nested loops
    #pragma omp parallel
    {
        // Create thread-local heaps to avoid contention
        SwapHeap *local_heap = create_swap_heap(1000);

        #pragma omp for schedule(dynamic)
        for (int node = 1; node <= NUM_NODES; node++) {
            if(interrupted != 0){ //Theoretically, it'll just run as fast as it can till the end
               continue;
            }

            const int swapNodes[2] = {nodeM1, nodeM2};

            for (int i = 0; i < 2; i++) {
                // Calculate swap delta, which is thread-safe as it doesn't modify shared state
                int swapDelta = calculate_swap_delta(gm, gf, bestMapping, node, swapNodes[i]);

                // Use atomic updates to safely modify shared lookup array
                #pragma omp critical
                {
                    lookup[node * (NUM_NODES+1) + swapNodes[i]] = (short) swapDelta;
                    lookup[swapNodes[i] * (NUM_NODES+1) + node] = (short) swapDelta;
                }

                // Collect swaps with positive or zero delta
                if (swapNodes[i] != node && (swapDelta > 0 || (ACCEPT_ZERO_DELTA && swapDelta == 0))) {
                    insert_swap(local_heap, (Swap){node, swapNodes[i], swapDelta});
                }
            }
        }

        // Merge local heap into global heap with a critical section
        #pragma omp critical
        {
            for (int i = 0; i < local_heap->size; i++) {
                insert_swap(swapHeap, local_heap->data[i]);
            }
        }

        // Clean up local heap
        destroy_swap_heap(local_heap);
    }
}

void update_lookup_and_heap_from_selected_nodes(short *lookup, SwapHeap *swapHeap, Graph *gm, 
                                              Graph *gf, int *bestMapping, bool *selectedNodes) {
    empty_heap(swapHeap);

    int nodes_updated = 0;
    int nodes_to_update = 0;
    for(int i = 1; i <= NUM_NODES; i++){
        if(selectedNodes[i]) nodes_to_update++;
    }

    LOG_INFO("Nodes to update: %d", nodes_to_update);

    #pragma omp parallel
    {
        // Create thread-local heap to avoid contention
        SwapHeap *local_heap = create_swap_heap(1000);
        
        // First loop: each thread processes a subset of all nodes
        #pragma omp for schedule(dynamic)
        for (int node1 = 1; node1 <= NUM_NODES; node1++) {
            if(interrupted != 0){ //Again, theoretically it'll just run as fast as it can till the end
                continue;
            }
            // Only process if this node needs updating
            if (selectedNodes[node1]) {
                #pragma omp critical
                {
                    nodes_updated++;
                    if(nodes_updated % 20 == 0){
                        LOG_INFO("Nodes updated: %d out of %d", nodes_updated, nodes_to_update);
                    }
                }
                // For each selected node, we need to calculate its interaction with all other nodes
                for (int node2 = 1; node2 <= NUM_NODES; node2++) {
                    if (node1 != node2) {
                        short swapDelta = (short) calculate_swap_delta(gm, gf, bestMapping, node1, node2);
                        
                        // Update lookup table - needs synchronization
                        #pragma omp critical
                        {
                            lookup[node1 * (NUM_NODES+1) + node2] = swapDelta;
                            lookup[node2 * (NUM_NODES+1) + node1] = swapDelta;
                        }
                        
                        // Add to heap if it's a beneficial swap
                        if (node1 > node2 && (swapDelta > 0 || (ACCEPT_ZERO_DELTA && swapDelta == 0))) {
                            insert_swap(local_heap, (Swap){node1, node2, swapDelta});
                        }
                    }
                }
            }
        }
        
        // Merge thread-local heap into global heap
        #pragma omp critical
        {
            for (int i = 0; i < local_heap->size; i++) {
                insert_swap(swapHeap, local_heap->data[i]);
            }
        }
        
        // Clean up local heap
        destroy_swap_heap(local_heap);
    }

    if(interrupted == 0){
        reinsert_into_heap(swapHeap, lookup, true);
    }
}

short* create_score_lookup(Graph* gm, Graph* gf, int* mapping, short* lookup) {
    LOG_INFO("Creating score lookup using OpenMP...");
    // short* lookup = (short*)calloc(NUM_NODES * NUM_NODES, sizeof(short));
    
    const int maxCount = NUM_NODES * (NUM_NODES-1) / 2;
    volatile long progress = 0;  // Shared progress counter
    
    // Initialize diagonal elements first
    for (int i = 1; i <= NUM_NODES; i++) {
        lookup[i*NUM_NODES + i] = 0;
    }
    
    #pragma omp parallel
    {
        #pragma omp single
        LOG_INFO("Using %d threads", omp_get_num_threads());
        
        // Each thread will get its own start/end range
        #pragma omp for schedule(dynamic, 10000)
        for (int n = 0; n < maxCount; n++) {
            // Convert linear index n to (i,j) coordinates
            // For lower triangle: n = (i * (i-1))/2 + j
            // Solve quadratic equation: i^2 - i - 2n = 0
            // int i = (int)((sqrt(1 + 8.0 * n) + 1.0) / 2.0);
            int i = (int)((1.0 + sqrt(1 + 8 * n)) * 0.5);
            int j = n - (i * (i-1))/2;
            
            short delta = (short) calculate_swap_delta(gm, gf, mapping, i+1, j+1);
            // short delta = calculate_swap_delta(gm, gf, mapping, i+1, j+1);
            lookup[(i+1)*NUM_NODES + j+1] = delta;
            // lookup[j*NUM_NODES + i] = (short) calculate_swap_delta(gm, gf, mapping, j+1, i+1);
            lookup[(j+1)*NUM_NODES + i+1] = delta;

            
            #pragma omp atomic
            progress++;
            
            if ((progress & 0x3FFFF) == 0) {
                #pragma omp critical
                {
                    print_progress(progress, maxCount, "Creating score lookup");
                }
            }
        }
    }
    
    return lookup;
}

// Random number between 0 and 1
static inline double random_double(struct xoshiro256ss_state *state){
    return ((double) xoshiro256ss(state)) / (double) UINT64_MAX;
}

static inline bool acceptProbabilisticSwap(int delta, double temperature, struct xoshiro256ss_state *state) { 
    return random_double(state) < exp((double) delta / temperature); 
}

int probabilistic_swaps(short *lookup, SwapHeap *swapHeap, Graph *gm,
                         Graph *gf, int *bestMapping, int score, int numSwaps, double temperature, bool acceptZeroDelta, xoshiro256ss_state *state) {
    
    LOG_INFO("Performing %d probabilistic swaps with temperature %f", numSwaps, temperature);
    int swaps = 0;
    int evaluatedSwaps = 0;
    int newScore = score;
    while (swaps < numSwaps) {
        if(evaluatedSwaps % 1000000 == 0 && interrupted != 0){
            return newScore;
        }
        evaluatedSwaps++;
        int nodeM1 = (int) (xoshiro256ss(state) % (NUM_NODES-1)) + 1;
        int nodeM2 = (int) (xoshiro256ss(state) % NUM_NODES) + 1;
        if (nodeM1 == nodeM2) nodeM1++;

        int delta = lookup[nodeM1 * (NUM_NODES+1) + nodeM2];

        if (delta == 0 && !acceptZeroDelta) continue;

        if(delta > 0 || (acceptZeroDelta && delta == 0) || acceptProbabilisticSwap(delta, temperature, state)) {
            swaps++;
            
            int temp = bestMapping[nodeM1];
            bestMapping[nodeM1] = bestMapping[nodeM2];
            bestMapping[nodeM2] = temp;
            
            newScore += delta;

            invalidate_swaps(swapHeap, nodeM1);
            invalidate_swaps(swapHeap, nodeM2);

            update_lookup_and_heap(lookup, swapHeap, gm, gf, bestMapping, nodeM1, nodeM2);
        }
    }

    LOG_INFO("Evaluated %d swaps, performed %d swaps", evaluatedSwaps, numSwaps);
    LOG_INFO("New score: %d, actual new score: %d, total delta: %d", newScore, calculate_alignment_score(gm, gf, bestMapping), newScore - score);

    return newScore;
}

int simulated_annealing(short *lookup, SwapHeap *swapHeap, Graph *gm,
                         Graph *gf, int *inMapping, int score, int iterations, double tmin, double tmax, xoshiro256ss_state *state) {
    LOG_INFO("Performing %d simulated annealing iterations from T=%.2lf to T=%.2lf", iterations, tmax, tmin);
    
    double t_factor = log(tmin / tmax);

    int curScore = score;
    int swaps = 0;

    int* bestMapping = calloc(NUM_NODES + 1, sizeof(int));
    int bestScore = score;

    memcpy(bestMapping, inMapping, (NUM_NODES + 1) * sizeof(int));

    for(int i = 0; i < iterations; i++){
        
        double temperature = tmax * exp(t_factor * i / (double) iterations);
        
        if(i % 10000000 == 0){
            LOG_INFO("Iteration %d of %d (%.2f%% complete, T=%.2lf). Performed %d swaps. Current score: %d, best score: %d", i, iterations, 100.0 * i / iterations, temperature, swaps, curScore, bestScore);
            if(interrupted != 0) return bestScore;
        }

        int nodeM1 = (int) (xoshiro256ss(state) % (NUM_NODES-1)) + 1;
        int nodeM2 = (int) (xoshiro256ss(state) % NUM_NODES) + 1;
        if (nodeM1 == nodeM2) nodeM1++;

        if(nodeM1 == 0 || nodeM2 == 0){
            LOG_DEBUG("Swap with node 0 in iteration %d of simulated annealing", i);
            continue;
        }

        if(nodeM1 == nodeM2){
            LOG_DEBUG("Swap with same node in iteration %d of simulated annealing", i);
            continue;
        }

        int delta = lookup[nodeM1 * (NUM_NODES+1) + nodeM2];

        if (delta >= 0 || acceptProbabilisticSwap(delta, temperature, state) ) {
            swaps++;

            SWAP(inMapping[nodeM1], inMapping[nodeM2], int);
            
            curScore += delta;

            invalidate_swaps(swapHeap, nodeM1);
            invalidate_swaps(swapHeap, nodeM2);

            update_lookup_and_heap(lookup, swapHeap, gm, gf, inMapping, nodeM1, nodeM2);
            
            if(curScore > bestScore){
                bestScore = curScore;
                memcpy(bestMapping, inMapping, (NUM_NODES + 1) * sizeof(int));
            }
        }
    }

    LOG_INFO("Performed %d swaps", swaps);
    LOG_INFO("New score: %d, actual new score: %d, total delta: %d", curScore, calculate_alignment_score(gm, gf, inMapping), curScore - score);

    return calculate_alignment_score(gm, gf, inMapping);
}

int parallelized_simulated_annealing(short *lookup, SwapHeap *swapHeap, Graph *gm,
                         Graph *gf, int *inMapping, int score, int iterations, double tmin, double tmax, xoshiro256ss_state *state) {
    LOG_INFO("Performing %d parallel simulated annealing iterations from T=%.2lf to T=%.2lf", iterations, tmax, tmin);
    
    double t_factor = log(tmin / tmax);
    int num_threads = omp_get_max_threads();

    // Allocate memory for each thread's mapping and score
    int **threadMappings = malloc(num_threads * sizeof(int*));
    bool **threadChanged = malloc(num_threads * sizeof(bool*));  // Track which nodes have been modified
    int *threadScores = calloc(num_threads, sizeof(int));
    
    // Keep track of best solution for each thread
    int *bestThreadScores = calloc(num_threads, sizeof(int));
    int **bestThreadMappings = malloc(num_threads * sizeof(int*));
    
    // Initialize thread-local data
    for(int i = 0; i < num_threads; i++) {
        threadMappings[i] = malloc((NUM_NODES + 1) * sizeof(int));
        bestThreadMappings[i] = malloc((NUM_NODES + 1) * sizeof(int));
        threadChanged[i] = calloc(NUM_NODES + 1, sizeof(bool));  // Initialize all nodes as unchanged
        memcpy(threadMappings[i], inMapping, (NUM_NODES + 1) * sizeof(int));
        memcpy(bestThreadMappings[i], inMapping, (NUM_NODES + 1) * sizeof(int));
        threadScores[i] = score;
        bestThreadScores[i] = score;
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        // Create thread-local RNG state
        xoshiro256ss_state thread_state = *state;
        thread_state.s[0] ^= thread_id; // Ensure different seeds for each thread
        
        for(int i = 0; i < iterations; i++) {
            double temperature = tmax * exp(t_factor * i / (double) iterations);
            
            if(i % 10000000 == 0) {
                LOG_INFO("Iteration %d of %d (%.2f%% complete, T=%.2lf). Thread %d score: %d", 
                        i, iterations, 100.0 * i / iterations, temperature, thread_id, threadScores[thread_id]);
                if(interrupted != 0) break;
            }

            int nodeM1 = (int) (xoshiro256ss(&thread_state) % (NUM_NODES-1)) + 1;
            int nodeM2 = (int) (xoshiro256ss(&thread_state) % NUM_NODES) + 1;
            if (nodeM1 == nodeM2) nodeM1++;

            // Check if either node has been modified and calculate delta accordingly
            int delta;
            if(threadChanged[thread_id][nodeM1] || threadChanged[thread_id][nodeM2]) {
                // If either node has been modified, we need to calculate the delta directly
                delta = calculate_swap_delta(gm, gf, threadMappings[thread_id], nodeM1, nodeM2);
            } else {
                // If neither node has been modified, we can use the lookup table
                delta = lookup[nodeM1 * (NUM_NODES+1) + nodeM2];
            }

            if (delta >= 0 || acceptProbabilisticSwap(delta, temperature, &thread_state)) {
                // Perform swap in thread-local mapping
                SWAP(threadMappings[thread_id][nodeM1], threadMappings[thread_id][nodeM2], int);
                threadScores[thread_id] += delta;
                
                // Mark these nodes as changed
                threadChanged[thread_id][nodeM1] = true;
                threadChanged[thread_id][nodeM2] = true;

                if(threadScores[thread_id] > bestThreadScores[thread_id]) {
                    bestThreadScores[thread_id] = threadScores[thread_id];
                    memcpy(bestThreadMappings[thread_id], threadMappings[thread_id], 
                          (NUM_NODES + 1) * sizeof(int));
                }
            }
        }
    }

    // Find the best solution among all threads
    int bestScore = bestThreadScores[0];
    int bestThread = 0;
    for(int i = 1; i < num_threads; i++) {
        int actualScore = calculate_alignment_score(gm, gf, bestThreadMappings[i]);
        threadScores[i] = actualScore;
        if(actualScore > bestScore) {
            bestScore = actualScore;
            bestThread = i;
        }
    }

    // Copy best solution back to input mapping
    memcpy(inMapping, threadMappings[bestThread], (NUM_NODES + 1) * sizeof(int));
    
    if(interrupted == 0){
        // Update the lookup table and heap for the changed nodes from the best thread
        update_lookup_and_heap_from_selected_nodes(lookup, swapHeap, gm, gf, threadMappings[bestThread], threadChanged[bestThread]);
    }
    
    // Clean up
    for(int i = 0; i < num_threads; i++) {
        free(threadMappings[i]);
        free(threadChanged[i]);
        free(bestThreadMappings[i]);
    }
    free(threadMappings);
    free(bestThreadMappings);
    free(threadChanged);
    free(threadScores);
    free(bestThreadScores);

    return bestScore;
}
int *optimize_mapping(Graph *gm, Graph *gf, int *inMapping, short *lookup,
                     int *outMapping, SwapHeap *swapHeap, const char* outMappingFile, const char* outLookupFile) {
    int score = calculate_alignment_score(gm, gf, inMapping);
    int *bestMapping = calloc(NUM_NODES + 1, sizeof(int));
    memcpy(bestMapping, inMapping, (NUM_NODES + 1) * sizeof(int));

    int bestScore = score;
    bool changed = false;

    int lastImproved = 0;

    xoshiro256ss_state state;
    xoshiro256ss_init(&state, time(NULL));

    Swap prevSwap = {-1,-1};

    for (int i = 1; i <= ITERATIONS; i++) {
        if(interrupted != 0){ 
            break;
        }
        if(i % UPDATE_INTERVAL == 0){
            time_t now = time(NULL);
            LOG_DEBUG("Score: %d at time %s", score, ctime(&now));
        }

        if(i % REINSERT_INTERVAL == 0){
            LOG_INFO("Reinserting into heap at iteration %d", i);
            reinsert_into_heap(swapHeap, lookup, true);
            LOG_INFO("Finished reinsertion");
        } 

        bool atInterval = i % PROBABILISTIC_INTERVAL == 0 || i % SIMULATED_ANNEALING_INTERVAL == 0;
        if( (PROBABILISTIC || SIMULATED_ANNEALING) && atInterval && (i - lastImproved) > 1500 && no_positive_deltas(swapHeap)){
             if(PROBABILISTIC){
                LOG_INFO("Performing probabilistic swaps at iteration %d", i);
                score = probabilistic_swaps(lookup, swapHeap, gm, gf, inMapping, score, PROBABILISTIC_SWAPS, PROBABILISTIC_TEMPERATURE, false, &state);
            }
            if(SIMULATED_ANNEALING && i % SIMULATED_ANNEALING_INTERVAL == 0){
                LOG_INFO("Performing simulated annealing at iteration %d", i);
                score = parallelized_simulated_annealing(lookup, swapHeap, gm, gf, inMapping, calculate_alignment_score(gm, gf, inMapping), SIMULATED_ANNEALING_ITERATIONS, SIMULATED_ANNEALING_TMIN, SIMULATED_ANNEALING_TMAX, &state);
                empty_heap(swapHeap);
                reinsert_into_heap(swapHeap, lookup, true);
            }
            reinsert_into_heap(swapHeap, lookup, true);
        }
        
        if(HUNGARIAN && i % HUNGARIAN_INTERVAL == 0){
            LOG_INFO("Performing hungarian at iteration %d", i);
            // int* k_schedule = calloc(HUNGARIAN_ITERS, sizeof(int));
            // for(int j = 0; j < HUNGARIAN_ITERS; j++) {
            //     // // first third of the schedule
            //     // if(j < HUNGARIAN_ITERS / 3) k_schedule[j] = 600;
            //     // // second third of the schedule
            //     // else if(j < 2 * HUNGARIAN_ITERS / 3) k_schedule[j] = 700;
            //     // // last third of the schedule
            //     // else k_schedule[j] = 800;
            //     k_schedule[j] = 1000;
            // }
            hungarian_out hungarianResult = optimize_hungarian(gm, gf, inMapping, calculate_alignment_score(gm, gf, inMapping), HUNGARIAN_ITERS);
            // free(k_schedule);

            memcpy(inMapping, hungarianResult.mapping, (NUM_NODES + 1) * sizeof(int));
            free(hungarianResult.mapping);
            time_t start = time(NULL);
            LOG_INFO("Updating lookup and heap from selected nodes at iteration %d", i);
            int num_updated = 0;
            for(int i = 1; i <= NUM_NODES; i++){
                if(inMapping[i] != 0){
                    num_updated++;
                }
            }
            // if(num_updated <= 3000){
            update_lookup_and_heap_from_selected_nodes(lookup, swapHeap, gm, gf, inMapping, hungarianResult.updated_nodes);
            // }else{
            //     create_score_lookup(gm, gf, inMapping, lookup);
            // }
            // empty_heap(swapHeap);
            // create_score_lookup(gm, gf, inMapping, lookup);
            reinsert_into_heap(swapHeap, lookup, true);
            LOG_INFO("Reinserted into heap at iteration %d, size: %d", i, swapHeap->size);

            LOG_INFO("Finished hungarian at iteration %d in %.2f seconds", i, difftime(time(NULL), start));
            free(hungarianResult.updated_nodes);
            score = calculate_alignment_score(gm, gf, inMapping);

            LOG_DEBUG("Hungarian score: %d", score);
            
            if(score > bestScore){
                bestScore = score;
                changed = true;
                lastImproved = i;
                time_t now = time(NULL);
                LOG_DEBUG("New best score: %d at time %s", bestScore, ctime(&now));

                memcpy(bestMapping, inMapping, (NUM_NODES + 1) * sizeof(int));

                save_mapping(outMappingFile, bestMapping);

            }

        }

        Swap bestSwap;

        while(!extract_max(swapHeap, &bestSwap) || (bestSwap.nodeM1 == prevSwap.nodeM1 && bestSwap.nodeM2 == prevSwap.nodeM2) || (bestSwap.nodeM1 == prevSwap.nodeM2 && bestSwap.nodeM2 == prevSwap.nodeM1)){ 
            // LOG_ERROR("Failed to extract swap from heap at iteration %d; most likely heap is empty", i);
            // break;
            LOG_INFO("Reinserting into heap at iteration %d", i);
            reinsert_into_heap(swapHeap, lookup, true);
        }

        int nodeM1 = bestSwap.nodeM1;
        int nodeM2 = bestSwap.nodeM2;
        int delta = bestSwap.delta;

        prevSwap.nodeM1 = nodeM1;
        prevSwap.nodeM2 = nodeM2;

        if(nodeM1 == 0 || nodeM2 == 0){
            LOG_DEBUG("Swap with node 0 in iteration %d of optimize_mapping", i);
            invalidate_swaps(swapHeap, 0);
            update_lookup_and_heap(lookup, swapHeap, gm, gf, inMapping, 0, MAX(nodeM1, nodeM2));
            continue;
        }

        if(nodeM1 == nodeM2){
            LOG_DEBUG("Swap with same node (%d) in iteration %d of optimize_mapping", i, nodeM1);
            invalidate_swaps(swapHeap, nodeM1);
            update_lookup_and_heap(lookup, swapHeap, gm, gf, inMapping, nodeM1, nodeM2);
            continue;
        }

        if(i % 100 == 0){
            LOG_INFO("Best Swap: Score = %d\tDelta = %d\tNode 1 = %d\tNode 2 = %d, All zero delta: %s", score, delta, nodeM1, nodeM2, no_positive_deltas(swapHeap) ? "yes" : "no");
            invalidate_swaps(swapHeap, nodeM1);
            invalidate_swaps(swapHeap, nodeM2);
        }

        score += delta;

        SWAP(inMapping[nodeM1], inMapping[nodeM2], int);
        
        invalidate_swaps(swapHeap, nodeM1);
        invalidate_swaps(swapHeap, nodeM2);

        update_lookup_and_heap(lookup, swapHeap, gm, gf, inMapping, nodeM1, nodeM2);

        if(i % 100 == 0){
            int actual_score = calculate_alignment_score(gm, gf, inMapping);
            LOG_INFO("Iteration %d:\tScore = %d\tActual score: %d\tHeap size: %d", i, score, actual_score, swapHeap->size);
            if (actual_score != score){
                LOG_INFO("Score and actual score do not match, discrepancy of %d", actual_score - score);
            }

            score = actual_score;

            if(actual_score > bestScore){
                bestScore = actual_score;
                changed = true;
                lastImproved = i;
                
                time_t now = time(NULL);
                LOG_DEBUG("New best score: %d at time %s", bestScore, ctime(&now));

                memcpy(bestMapping, inMapping, (NUM_NODES + 1) * sizeof(int));

                save_mapping(outMappingFile, bestMapping);
            }
        }

        if (changed && i != 0 && i % SAVE_FREQUENCY == 0) {
            LOG_INFO("Saving mapping and lookup at iteration %d with score %d", i, bestScore);
            // save_lookup(outLookupFile, lookup);
            save_mapping(outMappingFile, bestMapping);

            changed = false;
        }
    }

    LOG_INFO("Final score: %d", score);
    int final_score = calculate_alignment_score(gm, gf, inMapping);
    LOG_INFO("Final actual score: %d", calculate_alignment_score(gm, gf, inMapping));

    if(final_score > bestScore){
        bestScore = final_score;
        memcpy(bestMapping, inMapping, (NUM_NODES + 1) * sizeof(int));
    }
    LOG_INFO("Saving final mapping and lookup with score %d", bestScore);

    if(interrupted == 0){
        save_lookup(outLookupFile, lookup);
    }
    save_mapping(outMappingFile, bestMapping);

    return bestMapping;
}

// Main function
int main(int argc, char *argv[]) {
    if (argc < 5) {
        LOG_ERROR(
            "Usage: %s <male graph> <female graph> <in mapping> <in lookup> <out mapping> <out lookup>",
            argv[0]);
        return 1;
    }

    omp_set_nested(1);  // Enable nested parallelism

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);
    // signal(SIGUSR2, sigint_handler);

    time_t total_start = time(NULL);
    LOG_INFO("Score Lookup Tool v1.0");
    LOG_INFO("Starting process with:");
    LOG_INFO("  - Male graph: %s", argv[1]);
    LOG_INFO("  - Female graph: %s", argv[2]);
    LOG_INFO("  - In Mapping: %s", argv[3]);
    LOG_INFO("  - In Lookup: %s", argv[4]);
    LOG_INFO("  - Output mapping: %s", argv[5]);
    LOG_INFO("  - Output Lookup: %s", argv[6]);

    Graph *gm, *gf = NULL;
    int *inMapping = NULL;
    int *bestMapping = NULL;
    short *lookup = NULL;
    SwapHeap *swapHeap = create_swap_heap(HEAP_CAPACITY);

    int EXIT_STATUS = 0;

    gm = load_graph_from_csv(argv[1]);
    if (!gm) {
        LOG_ERROR("Failed to load male graph");
        EXIT_STATUS = 1;
        goto cleanup;
    }

    gf = load_graph_from_csv(argv[2]);
    if (!gf) {
        LOG_ERROR("Failed to load female graph");
        EXIT_STATUS = 1;
        goto cleanup;
    }

    int max_node = MAX(get_max_node(gm), get_max_node(gf));

    inMapping = load_mapping(argv[3], max_node);
    if (!inMapping) {
        LOG_ERROR("Failed to load input mapping");
        EXIT_STATUS = 1;
        goto cleanup;
    }

    lookup = load_lookup(argv[4], swapHeap);
    // lookup = read_lookup_matrix(argv[4]);
    if (!lookup) {
        LOG_ERROR("Failed to load lookup file");
        EXIT_STATUS = 1;
        goto cleanup;
    }

    reinsert_into_heap(swapHeap, lookup, true);

    int initial_score = calculate_alignment_score(gm, gf, inMapping);
    LOG_INFO("Initial alignment score: %s", format_number(initial_score));

    bestMapping = optimize_mapping(gm, gf, inMapping, lookup, inMapping, swapHeap, argv[5], argv[6]);

    int final_score = calculate_alignment_score(gm, gf, bestMapping);

    time_t total_end = time(NULL);
    LOG_INFO("Process completed:");
    LOG_INFO("  - Initial score: %s", format_number(initial_score));
    LOG_INFO("  - Final score: %s", format_number(final_score));
    LOG_INFO("  - Improvement: %s", format_number(final_score - initial_score));
    LOG_INFO("  - Total time: %.1f minutes",
            difftime(total_end, total_start) / 60);

    cleanup:
        free_graph(gm);
        free_graph(gf);
        free(lookup);
        free(inMapping);
        free(bestMapping);
        destroy_swap_heap(swapHeap);

    return EXIT_STATUS;
}