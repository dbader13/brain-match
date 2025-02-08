#pragma once

#include "utils.h"
#include "parallel_hungarian.h"

typedef struct {
    int* mapping;
    bool* updated_nodes;
} hungarian_out;

// Calculate the delta (change in score) for swapping one node
int one_side_swap_delta(Graph* gm, Graph* gf, int* mapping, int node_m1, int node_m2);

/**
 * Find the optimal assignment of nodes from two graphs of size k, given
 * a cost matrix.
 */
int* optimal_assignment(int k, double** cost_matrix);

// Optimize mapping using Hungarian algorithm with subset selection
hungarian_out optimize_hungarian(Graph* gm, Graph* gf, int* initial_mapping, int current_score, int iters);