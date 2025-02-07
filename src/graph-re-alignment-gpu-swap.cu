#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
//#include <omp.h>

#include <cstdlib>  // for malloc
#include <cstdio>  // for io



// cuda_kernels.cuh
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

__global__ void calculateSwapDeltasKernel(
    const short* adj_matrix_m,
    const short* adj_matrix_f,
    const int* mapping,
    int num_nodes,
    int* deltas,
    int* node_pairs
);
__global__ void calculateSwapDeltasKernelQuick(
    const short* adj_matrix_m,
    const short* adj_matrix_f,
    const int* mapping,
    int num_nodes,
    const int* changed_vertices,  // Array of vertices that were swapped
    int num_changed,             // Number of changed vertices
    int* deltas,
    int* node_pairs
); 


#endif

// cuda_kernels.cu
//#include "cuda_kernels.cuh"


float Total_weight_score = 0.0f;

__global__ void calculateSwapDeltasKernel(
    const short* adj_matrix_m,
    const short* adj_matrix_f,
    const int* mapping,
    int num_nodes,
    int* deltas,
    int* node_pairs
) {


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int returnidx = idx;
    int num_pairs = (num_nodes * (num_nodes - 1)) / 2;

    if (idx >= num_pairs) return;

    // Convert linear index to vertex pair
    int node_m1 = 1;
    int temp = num_nodes - 1;
    while (idx >= temp) {
        idx -= temp;
        node_m1++;
        temp--;
    }
    int node_m2 = node_m1 + idx + 1;

    int matrix_dim = num_nodes + 1;  // Add this line
    int delta = 0;
    int node_f1 = mapping[node_m1];
    int node_f2 = mapping[node_m2];


    // Update all matrix indexing to use matrix_dim
    for (int i = 1; i <= num_nodes; i++) {
        if (i == node_m1 || i == node_m2) continue;


        // Outgoing edges from node_m1
        if (adj_matrix_m[node_m1 * matrix_dim + i] > 0) {
            int weight_m = adj_matrix_m[node_m1 * matrix_dim + i];
            int dst_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[node_f2 * matrix_dim + dst_f]) -
                    min(weight_m, adj_matrix_f[node_f1 * matrix_dim + dst_f]);
        }


        // Incoming edges to node_m1
        if (adj_matrix_m[i * matrix_dim + node_m1] > 0) {
            int weight_m = adj_matrix_m[i * matrix_dim + node_m1];
            int src_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f2]) - 
                    min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f1]);
        }
        
        // Outgoing edges from node_m2
        if (adj_matrix_m[node_m2 * matrix_dim + i] > 0) {
            int weight_m = adj_matrix_m[node_m2 * matrix_dim + i];
            int dst_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[node_f1 * matrix_dim + dst_f]) - 
                    min(weight_m, adj_matrix_f[node_f2 * matrix_dim + dst_f]);
        }
        
        // Incoming edges to node_m2
        if (adj_matrix_m[i * matrix_dim + node_m2] > 0) {
            int weight_m = adj_matrix_m[i * matrix_dim + node_m2];
            int src_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f1]) - 
                    min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f2]);
        }
    }
    
    // Handle direct edges between swapped nodes
    int m1_to_m2 = adj_matrix_m[node_m1 * matrix_dim + node_m2];
    if (m1_to_m2 > 0) {
        delta += min(m1_to_m2, adj_matrix_f[node_f2 * matrix_dim + node_f1]) - 
                min(m1_to_m2, adj_matrix_f[node_f1 * matrix_dim + node_f2]);
    }
    
    int m2_to_m1 = adj_matrix_m[node_m2 * matrix_dim + node_m1];
    if (m2_to_m1 > 0) {
        delta += min(m2_to_m1, adj_matrix_f[node_f1 * matrix_dim + node_f2]) - 
                min(m2_to_m1, adj_matrix_f[node_f2 * matrix_dim + node_f1]);
    }
    
    deltas[returnidx] = delta;
    node_pairs[returnidx * 2] = node_m1;
    node_pairs[returnidx * 2 + 1] = node_m2;


}

// Add new kernel parameters for changed vertices

__global__ void calculateSwapDeltasKernelQuick(
    const short* adj_matrix_m,
    const short* adj_matrix_f,
    const int* mapping,
    int num_nodes,
    const int* changed_vertices, 
    int num_changed,
    int* deltas,
    int* node_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int returnidx = idx;
    int num_unchanged = num_nodes - num_changed;
    int total_pairs = num_changed * num_unchanged + num_changed*(num_changed-1)/2;

    if (idx >= total_pairs) return;

    int matrix_dim = num_nodes + 1;  // Add this line
    int delta = 0;

    int node_m1,node_m2,node_f1,node_f2;
    int node_m1_idx,node_m2_idx;
    if (idx <num_changed*(num_changed-1)/2) {// both are changed vertices

        int node_m1_idx = 0;
        int temp = num_changed - 1;
        while (idx >= temp) {
            idx -= temp;
            node_m1_idx++;
            temp--;
        }
        int node_m2_idx = node_m1_idx + idx + 1;
	node_m1=changed_vertices[node_m1_idx];
	node_m2=changed_vertices[node_m2_idx];

        node_f1 = mapping[node_m1];
        node_f2 = mapping[node_m2];

    } else {
        idx -= num_changed*(num_changed-1)/2;
        // Get changed vertex
        node_m1_idx = idx / num_unchanged;
	if (node_m1_idx >= num_changed) {
		return;
	}
        node_m1 = changed_vertices[node_m1_idx];
	node_m2_idx = idx- node_m1_idx* num_unchanged;
        if (node_m2_idx >= num_unchanged ) {
            return;  // Beyond valid range
        }
        node_m2  = changed_vertices[num_changed+node_m2_idx];

        node_f1 = mapping[node_m1];
        node_f2 = mapping[node_m2];
    }

    // Rest of delta calculation remains the same
    for (int i = 1; i <= num_nodes; i++) {
        if (i == node_m1 || i == node_m2) continue;

        // Outgoing edges from node_m1
        if (adj_matrix_m[node_m1 * matrix_dim + i] > 0) {
            int weight_m = adj_matrix_m[node_m1 * matrix_dim + i];
            int dst_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[node_f2 * matrix_dim + dst_f]) -
                    min(weight_m, adj_matrix_f[node_f1 * matrix_dim + dst_f]);
        }


        // Incoming edges to node_m1
        if (adj_matrix_m[i * matrix_dim + node_m1] > 0) {
            int weight_m = adj_matrix_m[i * matrix_dim + node_m1];
            int src_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f2]) - 
                    min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f1]);
        }
        
        // Outgoing edges from node_m2
        if (adj_matrix_m[node_m2 * matrix_dim + i] > 0) {
            int weight_m = adj_matrix_m[node_m2 * matrix_dim + i];
            int dst_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[node_f1 * matrix_dim + dst_f]) - 
                    min(weight_m, adj_matrix_f[node_f2 * matrix_dim + dst_f]);
        }
        
        // Incoming edges to node_m2
        if (adj_matrix_m[i * matrix_dim + node_m2] > 0) {
            int weight_m = adj_matrix_m[i * matrix_dim + node_m2];
            int src_f = mapping[i];
            delta += min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f1]) - 
                    min(weight_m, adj_matrix_f[src_f * matrix_dim + node_f2]);
        }
    }
    
    // Handle direct edges between swapped nodes
    int m1_to_m2 = adj_matrix_m[node_m1 * matrix_dim + node_m2];
    if (m1_to_m2 > 0) {
        delta += min(m1_to_m2, adj_matrix_f[node_f2 * matrix_dim + node_f1]) - 
                min(m1_to_m2, adj_matrix_f[node_f1 * matrix_dim + node_f2]);
    }
    
    int m2_to_m1 = adj_matrix_m[node_m2 * matrix_dim + node_m1];
    if (m2_to_m1 > 0) {
        delta += min(m2_to_m1, adj_matrix_f[node_f1 * matrix_dim + node_f2]) - 
                min(m2_to_m1, adj_matrix_f[node_f2 * matrix_dim + node_f1]);
    }
    deltas[returnidx] = delta;
    node_pairs[returnidx * 2] = node_m1;
    node_pairs[returnidx * 2 + 1] = node_m2;
}




// Add these constants
#define SYNC_INTERVAL 1800  // Sync every 1 hour
#define TAG_SCORE 1
#define TAG_MAPPING 2
#define TAG_TERMINATE 3

#define SAVE_INTERVAL 4200
#define UPDATE_INTERVAL 4000
#define MAX_LINE_LENGTH 1024
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MAX_NODES 100000

const int NUM_NODES = 18524;

typedef enum {
    LOG_LEVEL_INFO = 0,
    LOG_LEVEL_DEBUG = 1,
    LOG_LEVEL_ERROR = 2
} LogLevel;

#define CURRENT_LOG_LEVEL LOG_LEVEL_INFO

#define LOG_ERROR(fmt, ...) \
    if (CURRENT_LOG_LEVEL <= LOG_LEVEL_ERROR) { \
        fprintf(stderr, "[ERROR] %s:%d: " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
    }

#define LOG_INFO(fmt, ...) \
    if (CURRENT_LOG_LEVEL <= LOG_LEVEL_INFO) { \
        printf("[INFO] " fmt "\n", ##__VA_ARGS__); \
    }

#define LOG_DEBUG(fmt, ...) \
    if (CURRENT_LOG_LEVEL <= LOG_LEVEL_DEBUG) { \
        printf("[DEBUG] %s: " fmt "\n", __func__, ##__VA_ARGS__); \
    }




void sortArray(int arr[], int size) {
    for(int i = 0; i < size-1; i++) {
        for(int j = 0; j < size-i-1; j++) {
            if(arr[j] > arr[j+1]) {
                // Swap elements
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}


// Structure definitions
typedef struct EdgeMap {
    int* to_nodes;
    int* weights;
    int count;
    int capacity;
} EdgeMap;

typedef struct Graph {
    EdgeMap* edges;
    EdgeMap* reverse_edges;
    short** adj_matrix;
    int node_capacity;
} Graph;

typedef struct NodeMetrics {
    int in_degree;
    int out_degree;
    int total_weight;
    double avg_in_weight;
    double avg_out_weight;
    int ordering_rank;
} NodeMetrics;


// Progress bar function
void print_progress(int current, int total, const char* prefix) {
    const int bar_width = 50;
    float progress = (float)current / total;
    int filled = (int)(bar_width * progress);
    
    printf("\r%s [", prefix);
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %.1f%%", progress * 100);
    fflush(stdout);
    if (current == total) printf("\n");
}

// Function to format numbers with commas
char* format_number(int num) {
    static char formatted[32];
    char temp[32];
    int i = 0, j = 0;
    
    sprintf(temp, "%d", num);
    int len = strlen(temp);
    
    while (len > 0) {
        if (i > 0 && i % 3 == 0) formatted[j++] = ',';
        formatted[j++] = temp[len - 1];
        len--;
        i++;
    }
    formatted[j] = '\0';
    
    // Reverse the string
    for (i = 0; i < j/2; i++) {
        char t = formatted[i];
        formatted[i] = formatted[j-1-i];
        formatted[j-1-i] = t;
    }
    
    return formatted;
}

EdgeMap* new_edge_map() {
    EdgeMap* em = (EdgeMap*)malloc(sizeof(EdgeMap));
    if (!em) {
        LOG_ERROR("Failed to allocate EdgeMap");
        exit(1);
    }
    em->capacity = 100;
    em->count = 0;
    em->to_nodes =(int *) malloc(sizeof(int) * em->capacity);
    em->weights = (int *) malloc(sizeof(int) * em->capacity);
    if (!em->to_nodes || !em->weights) {
        LOG_ERROR("Failed to allocate EdgeMap arrays");
        exit(1);
    }
    return em;
}

Graph* new_graph() {
    Graph* g =(Graph *) malloc(sizeof(Graph));
    if (!g) {
        LOG_ERROR("Failed to allocate Graph");
        exit(1);
    }
    g->edges = NULL;
    g->reverse_edges = NULL;
    g->adj_matrix = NULL;
    g->adj_matrix = (short**)malloc((NUM_NODES+1) * sizeof(short*));
    for (int i = 0; i <= NUM_NODES; ++i) {
        g->adj_matrix[i] = (short*)calloc((NUM_NODES+1), sizeof(short)); // Initialize to 0
    }

    //g->nodes = malloc(sizeof(int) * 10000);
    //if (!g->nodes) {
    //    LOG_ERROR("Failed to allocate nodes array");
    //    exit(1);
    //}
    //g->node_count = 0;
    g->node_capacity = 10000;
    return g;
}

void add_to_edge_map(EdgeMap* em, int to, int weight) {
    if (em->count >= em->capacity) {
        em->capacity *= 2;
        int* new_to_nodes = (int *) realloc(em->to_nodes, sizeof(int) * em->capacity);
        int* new_weights = (int *) realloc(em->weights, sizeof(int) * em->capacity);
        if (!new_to_nodes || !new_weights) {
            LOG_ERROR("Failed to reallocate EdgeMap arrays");
            exit(1);
        }
        em->to_nodes = new_to_nodes;
        em->weights = new_weights;
    }
    em->to_nodes[em->count] = to;
    em->weights[em->count] = weight;
    em->count++;
}

//void add_node(Graph* g, int node) {
//    for (int i = 0; i < g->node_count; i++) {
//        if (g->nodes[i] == node) return;
//    }
//    
//    if (g->node_count >= g->node_capacity) {
//        g->node_capacity *= 2;
//        int* new_nodes = realloc(g->nodes, sizeof(int) * g->node_capacity);
//        if (!new_nodes) {
//            LOG_ERROR("Failed to reallocate nodes array");
//            exit(1);
//        }
//        g->nodes = new_nodes;
//    }
//    g->nodes[g->node_count++] = node;
//}

void add_edge(Graph* g, int from, int to, int weight) {
  //    LOG_DEBUG("Adding edge: %d -> %d (weight: %d)", from, to, weight);
    g->adj_matrix[from][to] = weight;
    if (g->edges == NULL) {
        g->edges = (EdgeMap*) calloc(MAX_NODES, sizeof(EdgeMap));
        g->reverse_edges = (EdgeMap*)calloc(MAX_NODES, sizeof(EdgeMap));
        if (!g->edges || !g->reverse_edges) {
            LOG_ERROR("Failed to allocate edges arrays");
            exit(1);
        }
    }
    
    if (g->edges[from].count == 0) {
        g->edges[from] = *new_edge_map();
    }
    if (g->reverse_edges[to].count == 0) {
        g->reverse_edges[to] = *new_edge_map();
    }
    
    add_to_edge_map(&g->edges[from], to, weight);
    add_to_edge_map(&g->reverse_edges[to], from, weight);
}

int get_weight(Graph* g, int from, int to) {
    return g->adj_matrix[from][to];
    //for (int i = 0; i < g->edges[from].count; i++) {
    //    if (g->edges[from].to_nodes[i] == to) {
    //        return g->edges[from].weights[i];
    //    }
    //}
    //return 0;
}

int* read_ordering(const char* filename, int max_node) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Error opening file: %s", filename);
        exit(1);
    }

    int* ordering = (int *) calloc(max_node + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    
    LOG_INFO("Reading ordering from %s", filename);
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int node_id, order;
        sscanf(line, "%d,%d", &node_id, &order);
        ordering[node_id] = order;
    }
    
    fclose(file);
    return ordering;
}

NodeMetrics* calculate_node_metrics(Graph* g, const char* ordering_path) {
    int max_node = NUM_NODES;
    //for (int i = 0; i < g->node_count; i++) {
    //    if (g->nodes[i] > max_node) max_node = g->nodes[i];
    //}

    LOG_INFO("Calculating node metrics");
    
    int* ordering = (int *) read_ordering(ordering_path, max_node);
    NodeMetrics* metrics = ( NodeMetrics*) calloc(max_node + 1, sizeof(NodeMetrics));
    
    for (int node=1; node<=NUM_NODES; node++) {
        //int node = g->nodes[i];
        NodeMetrics* m = &metrics[node];
        
        m->out_degree = g->edges[node].count;
        for (int j = 0; j < g->edges[node].count; j++) {
            m->total_weight += g->edges[node].weights[j];
        }
        
        m->in_degree = g->reverse_edges[node].count;
        for (int j = 0; j < g->reverse_edges[node].count; j++) {
            m->total_weight += g->reverse_edges[node].weights[j];
        }
        
        if (m->out_degree > 0) {
            m->avg_out_weight = (double)m->total_weight / m->out_degree;
        }
        if (m->in_degree > 0) {
            m->avg_in_weight = (double)m->total_weight / m->in_degree;
        }
        
        m->ordering_rank = ordering[node];
    }
    
    free(ordering);
    return metrics;
}

double calculate_node_similarity(NodeMetrics m1, NodeMetrics m2) {
    double score = 5 * fabs(m1.in_degree - m2.in_degree) +
                   5 * fabs(m1.out_degree - m2.out_degree) +
                   fabs(m1.avg_in_weight - m2.avg_in_weight) +
                   fabs(m1.avg_out_weight - m2.avg_out_weight);
    
    double ordering_sim = fabs(m1.ordering_rank - m2.ordering_rank);
    score = 0.7 * score + 0.3 * ordering_sim;
    
    return -score;
}

int calculate_alignment_score(Graph* gm, Graph* gf, int* mapping) {
    int score = 0;
    
    for (int src_m = 1; src_m <= NUM_NODES; src_m++) {
        //int src_m = gm->i;
        for (int j = 0; j < gm->edges[src_m].count; j++) {
            int dst_m = gm->edges[src_m].to_nodes[j];
            int weight_m = gm->edges[src_m].weights[j];
            int src_f = mapping[src_m];
            int dst_f = mapping[dst_m];
            score += MIN(weight_m, gf->adj_matrix[src_f][dst_f]);
        }
    }
    
    return score;
}

void validate_mapping_changes(int* old_mapping, int* new_mapping, int max_node,
                            int node_m1, int node_m2) {
    for (int i = 1; i <= max_node; i++) {
        if (i != node_m1 && i != node_m2) {
            if (old_mapping[i] != new_mapping[i]) {
                LOG_ERROR("Unexpected mapping change for node %d: %d -> %d",
                         i, old_mapping[i], new_mapping[i]);
            }
        }
    }
}

int calculate_swap_delta(Graph* gm, Graph* gf, int* mapping, int node_m1, int node_m2) {
    int node_f1 = mapping[node_m1];
    int node_f2 = mapping[node_m2];
    int delta = 0;
    
    // Handle outgoing edges from node_m1
    for (int i = 0; i < gm->edges[node_m1].count; i++) {
        int dst_m = gm->edges[node_m1].to_nodes[i];
        if (dst_m == node_m2) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->edges[node_m1].weights[i];
        int dst_f = mapping[dst_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f1][dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f2][dst_f]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle incoming edges to node_m1
    for (int i = 0; i < gm->reverse_edges[node_m1].count; i++) {
        int src_m = gm->reverse_edges[node_m1].to_nodes[i];
        if (src_m == node_m2) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->reverse_edges[node_m1].weights[i];
        int src_f = mapping[src_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f1]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f2]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle outgoing edges from node_m2
    for (int i = 0; i < gm->edges[node_m2].count; i++) {
        int dst_m = gm->edges[node_m2].to_nodes[i];
        if (dst_m == node_m1) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->edges[node_m2].weights[i];
        int dst_f = mapping[dst_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f2][dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f1][dst_f]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle incoming edges to node_m2
    for (int i = 0; i < gm->reverse_edges[node_m2].count; i++) {
        int src_m = gm->reverse_edges[node_m2].to_nodes[i];
        if (src_m == node_m1) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->reverse_edges[node_m2].weights[i];
        int src_f = mapping[src_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f2]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f1]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle direct edges between the swapped nodes
    // From m1 to m2
    int m1_to_m2 = gm->adj_matrix[node_m1][node_m2];
    if (m1_to_m2 > 0) {
        int old_weight = MIN(m1_to_m2, gf->adj_matrix[node_f1][node_f2]);
        int new_weight = MIN(m1_to_m2, gf->adj_matrix[node_f2][node_f1]);
        delta += new_weight - old_weight;
    }
    
    // From m2 to m1
    int m2_to_m1 = gm->adj_matrix[node_m2][node_m1];
    if (m2_to_m1 > 0) {
        int old_weight = MIN(m2_to_m1, gf->adj_matrix[node_f2][node_f1]);
        int new_weight = MIN(m2_to_m1, gf->adj_matrix[node_f1][node_f2]);
        delta += new_weight - old_weight;
    }
    
    return delta;
}

void write_mapping(const char* filename, int* mapping, int max_node) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        LOG_ERROR("Error creating file: %s", filename);
        exit(1);
    }
    
    fprintf(file, "Male Node ID,Female Node ID\n");
    for (int i = 1; i <= max_node; i++) {
        if (mapping[i] != 0) {
            fprintf(file, "m%d,f%d\n", i, mapping[i]);
        }
    }
    
    fclose(file);
}

// Function to load benchmark mapping from CSV
int* load_benchmark_mapping(const char* filename, int max_node) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filename);
        return NULL;
    }
    
    int* mapping = (int *) calloc(max_node + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    int count = 0;
    
    // Try to get expected score from filename
    int expected_score = 0;
    const char* underscore = strrchr(filename, '_');
    if (underscore) {
        expected_score = atoi(underscore + 1);
        LOG_INFO("Expected score from filename: %d", expected_score);
    }
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    
    LOG_INFO("Loading benchmark mapping from %s", filename);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int male_id, female_id;
        
        // Try direct integer format first
        if (sscanf(line, "%d,%d", &male_id, &female_id) == 2) {
            mapping[male_id] = female_id;
            count++;
        } else {
            // Try format with prefixes
            char male_str[20], female_str[20];
            if (sscanf(line, "%[^,],%s", male_str, female_str) == 2) {
                male_id = atoi(male_str + (male_str[0] == 'm' ? 1 : 0));
                female_id = atoi(female_str + (female_str[0] == 'f' ? 1 : 0));
                mapping[male_id] = female_id;
                count++;
            }
        }
    }
    
    //LOG_INFO("Loaded %s mappings", format_number(count));
    
    fclose(file);
    return mapping;
}

// Save intermediate mapping with verification
void save_intermediate_mapping(const char* filename, int* mapping, int max_node, 
                             Graph* gm, Graph* gf, int current_score) {
    write_mapping(filename, mapping, max_node);
    
    // Verify written mapping
    int* verification = load_benchmark_mapping(filename, max_node);
    if (verification) {
        int verify_score = calculate_alignment_score(gm, gf, verification);
        if (verify_score != current_score) {
            LOG_ERROR("Score mismatch - internal: %d, written: %d", 
                     current_score, verify_score);
        }
        free(verification);
    }
}


// Helper function to calculate vertex scores based on influence
void calculate_vertex_scores(Graph* gm, float* scores, int n) {
    // Initialize scores array
    for (int i = 1; i <= n; i++) {
        scores[i] = 0.0f;
        
        // Calculate score based on out-degree and edge weights
        float out_weight = 0.0f;
        for (int j = 1; j <= n; j++) {
            if (gm->adj_matrix[i][j] > 0) {
                out_weight += gm->adj_matrix[i][j];
            }
        }
        
        // Calculate score based on in-degree and edge weights
        float in_weight = 0.0f;
        for (int j = 1; j <= n; j++) {
            if (gm->adj_matrix[j][i] > 0) {
                in_weight += gm->adj_matrix[j][i];
            }
        }
        
        // Score combines both out and in influence with preference to out-degree
        scores[i] = 0.7f * out_weight + 0.3f * in_weight;
    }
}

// Helper function to select vertices based on probability proportional to their scores
int select_vertex_by_score(float* scores, int* used, int n, float total_score) {
    float r = (float)rand() / RAND_MAX * total_score;
    float cumsum = 0.0f;
    
    for (int i = 1; i <= n; i++) {
        if (!used[i]) {
            cumsum += scores[i];
            if (cumsum >= r) {
                return i;
            }
        }
    }
    
    // Fallback to ensure we select something
    for (int i = 1; i <= n; i++) {
        if (!used[i]) {
            return i;
        }
    }
    return 1;  // Should never reach here if there are unused vertices
}

void random_swap_k_vertices(int* mapping, int n, int k, int seed,
                          int* changed_vertices, int* num_changed,
                          Graph* gm) {  // Add Graph* parameter
    // Create array for tracking selected vertices
    int* selected = (int*)malloc(2*k * sizeof(int));
    int* used = (int*)calloc(n + 1, sizeof(int));  // Track used positions
    float* vertex_scores = (float*)calloc(n + 1, sizeof(float));
    
    if (k*2 > n) {
        LOG_ERROR("K is too big : %d", k);
        free(selected);
        free(used);
        free(vertex_scores);
        return;
    }
    

    // Calculate vertex scores based on graph structure
    calculate_vertex_scores(gm, vertex_scores ,n);
    float total_score=0;
    // Calculate total score for normalization
    for (int i = 1; i <= n; i++) {
        total_score += vertex_scores[i];
    }

    // Randomly select vertices with probability proportional to their scores
    srand(time(NULL) + seed);
    int count = 0;
    // First k vertices selected based on scores (dominating set candidates)
    while (count < k) {
        int idx = select_vertex_by_score(vertex_scores, used, n, total_score);
        if (!used[idx]) {
            selected[count] = idx;
            used[idx] = 1;
            total_score -= vertex_scores[idx];  // Update remaining score
            count++;
        }
    }
    
    // Next k vertices selected with uniform probability
    while (count < 2*k) {
        int idx = rand() % n + 1;
        if (!used[idx]) {
            selected[count] = idx;
            used[idx] = 1;
            count++;
        }
    }
    
    // Store original values
    int* original_values = (int*)malloc(2*k * sizeof(int));
    for (int i = 0; i < 2*k; i++) {
        original_values[i] = mapping[selected[i]];
    }
    
    *num_changed = 0;
    for (int i = 0; i < k; i++) {
        // Store both vertices involved in each swap
        changed_vertices[(*num_changed)++] = selected[i];
        changed_vertices[(*num_changed)++] = selected[i+k];
        
        // Perform the swap
        mapping[selected[i]] = original_values[i+k];
        mapping[selected[i+k]] = original_values[i];
    }
    
    // Clean up
    free(selected);
    free(used);
    free(original_values);
    free(vertex_scores);
}







/*
void random_swap_k_vertices(int* mapping, int n, int k, int seed,
                          int* changed_vertices, int* num_changed) {
    // Create array for tracking selected vertices
    int* selected = (int*)malloc(2*k * sizeof(int));
    int* used = (int*)calloc(n, sizeof(int));  // Track used positions

    if (k*2>n) {
            LOG_ERROR("K is too big : %d",k); 
    } 
    // Randomly select k vertices
    int count = 0;
    srand(time(NULL)+seed);
    while (count < k*2) {
        int idx = rand() % (n+1);
	if (idx==0) {
		idx++;
	}
        if (!used[idx]) {
            selected[count] = idx;
            used[idx] = 1;
            count++;
        }
    }

    // Store original values
    int* original_values = (int*)malloc(2*k * sizeof(int));
    for (int i = 0; i <2* k; i++) {
        original_values[i] = mapping[selected[i]];

    }

    *num_changed = 0;
    for (int i = 0; i < k; i++) {
        // Store both vertices involved in each swap
        changed_vertices[(*num_changed)++] = selected[i];
        changed_vertices[(*num_changed)++] = selected[i+k];

        mapping[selected[i]] = original_values[i+k];
        mapping[selected[i+k]] = original_values[i];
    }

    // Clean up
    free(selected);
    free(used);
    free(original_values);
}

*/


// Add this function
void verify_matrix_copy(Graph* g, int max_node) {
    short* h_adj_matrix = (short*)malloc((max_node + 1) * (max_node + 1) * sizeof(short));

    // Count non-zero entries in original
    int orig_nonzero = 0;
    for (int i = 1; i <= max_node; i++) {
        for (int j = 1; j <= max_node; j++) {
            if (g->adj_matrix[i][j] > 0) {
                orig_nonzero++;
                printf("Found edge %d->%d weight=%d\n",
                       i, j, g->adj_matrix[i][j]);
            }
            h_adj_matrix[i * (max_node + 1) + j] = g->adj_matrix[i][j];
        }
    }

    // Verify copied matrix
    int copy_nonzero = 0;
    for (int i = 1; i <= max_node; i++) {
        for (int j = 1; j <= max_node; j++) {
            if (h_adj_matrix[i * (max_node + 1) + j] > 0) {
                copy_nonzero++;
            }
        }
    }

    printf("Original matrix had %d edges, copied matrix has %d edges\n",
           orig_nonzero, copy_nonzero);

    free(h_adj_matrix);
}





void    fillArray(int changed_vertices[],int num_changed,int max_node){
		    int* present = (int*)calloc(max_node, sizeof(int));

		    for(int i = 0; i < num_changed ; i++) {
		        present[changed_vertices[i]-1] = 1;
		    }

                    // Fill rest of array with missing numbers
                   int nextPos = num_changed;  // First empty position
                   for(int i=0;i<max_node;i++)   {
                        if(present[i]!=1) {
                                changed_vertices[nextPos++] = i+1;
		        }
		    }

		    free(present);
}

// Modified optimize_mapping function
int* optimize_mapping(Graph* gm, Graph* gf, int* initial_mapping, const char* out_path) {



    // Check if CUDA is already initialized
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA not initialized: %s", cudaGetErrorString(err));
        return NULL;
    }
    printf("Using CUDA device %d\n", device);
    // Print device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using CUDA device: %s\n", prop.name);



    int rank=0;
    int size=4;
    
    int max_node = NUM_NODES;
    int num_pairs = (max_node * (max_node - 1)) / 2;
    
    // Allocate and initialize mappings
    int* current_mapping = (int*)malloc((max_node + 1) * sizeof(int));
    int* best_mapping = (int*)malloc((max_node + 1) * sizeof(int));
    int* old_mapping = (int*)malloc((max_node + 1) * sizeof(int));
    memcpy(current_mapping, initial_mapping, (max_node + 1) * sizeof(int));
    memcpy(best_mapping, current_mapping, (max_node + 1) * sizeof(int));
    
    // Prepare GPU data structures
    short *d_adj_matrix_m, *d_adj_matrix_f;
    int *d_mapping, *d_deltas, *d_node_pairs;
    

    // Allocate GPU memory
    cudaMalloc(&d_adj_matrix_m, (max_node + 1) * (max_node + 1) * sizeof(short));
    cudaMalloc(&d_adj_matrix_f, (max_node + 1) * (max_node + 1) * sizeof(short));
    cudaMalloc(&d_mapping, (max_node + 1) * sizeof(int));
    cudaMalloc(&d_deltas, num_pairs * sizeof(int));
    cudaMalloc(&d_node_pairs, num_pairs * 2 * sizeof(int));
    
    // Convert adjacency matrices to linear arrays and copy to GPU
    short* h_adj_matrix_m = (short*)malloc((max_node + 1) * (max_node + 1) * sizeof(short));
    short* h_adj_matrix_f = (short*)malloc((max_node + 1) * (max_node + 1) * sizeof(short));
    
    for (int i = 1; i <= max_node; i++) {
        for (int j = 1; j <= max_node; j++) {
            h_adj_matrix_m[i * (max_node + 1) + j] = gm->adj_matrix[i][j];
            h_adj_matrix_f[i * (max_node + 1) + j] = gf->adj_matrix[i][j];
        }
    }
    




    cudaMemcpy(d_adj_matrix_m, h_adj_matrix_m, (max_node + 1) * (max_node + 1) * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_matrix_f, h_adj_matrix_f, (max_node + 1) * (max_node + 1) * sizeof(short), cudaMemcpyHostToDevice);
    


    // In optimize_mapping function:
    int* d_changed_vertices;
    cudaMalloc(&d_changed_vertices, max_node * sizeof(int));


    // Initialize other variables
    int current_score = calculate_alignment_score(gm, gf, current_mapping);
    int best_score = current_score;
    int improvements = 0;
    time_t start_time = time(NULL);
    time_t last_sync_time = start_time;
    int last = 0, pass = 0;
    int iter=0;
    // Track changed vertices
    int* changed_vertices = (int*)malloc( max_node * sizeof(int));
    int num_changed = 0;
    
    // Apply initial random perturbation
    //random_swap_k_vertices(current_mapping, max_node, 10 + rank, rank,
    //                     changed_vertices, &num_changed);
    //current_score = calculate_alignment_score(gm, gf, current_mapping);
    
    // Allocate host memory for results
    int* h_deltas = (int*)malloc(num_pairs * sizeof(int));
    int* h_node_pairs = (int*)malloc(num_pairs * 2 * sizeof(int));
    int  num_positives=0;
    int  num_zeros=0;
    int* positive_pairs = (int*)malloc(num_pairs * 2 * sizeof(int));

    int node_m1 ;
    int node_m2 ;
    int max_delta = 0;
    int best_pair_idx = -1;
    int pairs_per_changed ;
    int total_pairs ;
    int threadsPerBlock = 256;
    int numBlocks;  
    int cpu_delta;
    int num_pair ;



    // Main optimization loop
    while (true) {
        // Synchronization block
        time_t current_time = time(NULL);
        

        if (difftime(current_time, last_sync_time) >= SYNC_INTERVAL) {
              LOG_INFO("Process %d, enter synchronization the best score is %d", rank,best_score);
              int* benchmark_mapping = load_benchmark_mapping("../data/best.csv", max_node);
              if (!benchmark_mapping) {
                  LOG_ERROR("Failed to load benchmark mapping");
              }

              int benchmark_score = calculate_alignment_score(gm, gf, benchmark_mapping);
              if (benchmark_score >best_score) {
                      best_score=benchmark_score;
                      memcpy(best_mapping, benchmark_mapping, sizeof(int) * (max_node + 1));
              }
              free(benchmark_mapping);


              last_sync_time = current_time;
              LOG_INFO("Process %d, after synchronization and best score is %d", rank,best_score);

        }


        
            	// Inside the main loop, after random_swap_k_vertices:
       if  (max_delta == 0  ) {
                    // Random swap happened
	            improvements=0;
                    //num_pair = 3 + rand() % 2;
                    num_pair = 100 + rand() % 2;
    
                    if (iter-last<500  || (best_score-current_score)*1.0/5154247.0 >0.002) {
		        memcpy(current_mapping, best_mapping, (max_node + 1) * sizeof(int));
			    //printf("Perturbate best  in iter %d, last is %d, current score =%d, best score =%d\n",iter,last, current_score,best_score);
		    } else {
			    //printf("Perturbate current  in iter %d, last is %d, current score =%d, best score =%d\n",iter,last, current_score,best_score);
		    }
    

		    // Modified random_swap_k_vertices to return changed vertices
		    random_swap_k_vertices(current_mapping, max_node, num_pair, rank, 
                         changed_vertices, &num_changed,gm);


                    sortArray(changed_vertices,num_changed);

                    fillArray(changed_vertices,num_changed,max_node);
		    // Get score before random swap
                    current_score = calculate_alignment_score(gm, gf, current_mapping);

    
		    // Upload changed vertices to GPU
		    cudaMemcpy(d_changed_vertices, changed_vertices, 
		               max_node * sizeof(int), cudaMemcpyHostToDevice);
    
                    cudaMemcpy(d_mapping, current_mapping, (max_node + 1) * sizeof(int), cudaMemcpyHostToDevice);
		    // Calculate new number of pairs to check
		    total_pairs = num_changed * (num_changed-1)/2+ num_changed* (max_node-num_changed);
    
		    // Launch kernel with new parameters
		    threadsPerBlock = 256;
		    numBlocks = (total_pairs + threadsPerBlock - 1) / threadsPerBlock;
    

		    calculateSwapDeltasKernelQuick<<<numBlocks, threadsPerBlock>>>(
		        d_adj_matrix_m,
		        d_adj_matrix_f,
		        d_mapping,
		        max_node,
		        d_changed_vertices,
		        num_changed,
		        d_deltas,
		        d_node_pairs
		    );
	            cudaDeviceSynchronize();
                    iter++;
        	    current_time = time(NULL);
                    if (iter % 2400 ==0) { 
    			            LOG_INFO("Average time for once quick brute-force search is  %f s for iteration %d, running time =%d s", (current_time-start_time)/(iter*1.0),iter,current_time-start_time);
			            LOG_INFO("Current score is %d  best score is %d of rank %d, improvement =%d", current_score, best_score, rank,improvements);
		    }
        		// Copy results back to host
		    cudaMemcpy(h_deltas, d_deltas, total_pairs * sizeof(int), cudaMemcpyDeviceToHost);
		    cudaMemcpy(h_node_pairs, d_node_pairs, total_pairs * 2 * sizeof(int), cudaMemcpyDeviceToHost);
        
		        // Find best improvement
		    max_delta = 0;
		    best_pair_idx = -1;
                    num_positives=0;
		    num_zeros=0;
		    for (int i = 0; i < total_pairs; i++) {
		         if (h_deltas[i] > max_delta) {
		               max_delta = h_deltas[i];
		               best_pair_idx = i;
		          }
		         if (h_deltas[i] > 0 ) {
		               positive_pairs[num_positives*2]=h_node_pairs[i*2];
		               positive_pairs[num_positives*2+1]=h_node_pairs[i*2+1];
			       num_positives++;
			 }
		         if (h_deltas[i] == 0 ) {
		               positive_pairs[num_pairs * 2-2-num_zeros*2]=h_node_pairs[i*2];
		               positive_pairs[num_pairs * 2-2-num_zeros*2+1]=h_node_pairs[i*2+1];
			       num_zeros++;
			 }

		    }
                    int zerocheck=0;
		    while  ((max_delta > 0 || num_zeros>0) && (zerocheck<1000)) {
			    if (max_delta>0) {
				    //if (rand() %2 ==0 iter-last>120 && num_positives>0) {
				    if (rand() % 2 ==0 ) {
					    int select=rand() % num_positives;
			            	    node_m1 = positive_pairs[select * 2];
			                    node_m2 = positive_pairs[select * 2 + 1];
					    int tmpdelta=calculate_swap_delta(gm, gf, current_mapping, node_m1,node_m2);
					    //printf("Positive, random pair %d with delta %d and the max is %d in iter %d, current score is %d, best score is %d\n",select,tmpdelta,max_delta,iter,current_score,best_score);

				    } else {
			                    node_m1 = h_node_pairs[best_pair_idx * 2];
			                    node_m2 = h_node_pairs[best_pair_idx * 2 + 1];
					    //printf("Positive, best max delta is %d in iter %d, current score is %d, best score is %d\n",max_delta,iter,current_score,best_score);
				    }
				    zerocheck=0;
			    } else {
					    int select=rand() % num_zeros;
			            	    node_m1 = positive_pairs[num_pairs * 2-2-select * 2];
			                    node_m2 = positive_pairs[num_pairs * 2-2-select * 2 + 1];
					    zerocheck++;
					    //printf("Zero,select zero pair  %d in iter %d,current score= %d best score= %d\n",select,iter,current_score,best_score);
			    }
            
            		    // Perform swap
		            int temp = current_mapping[node_m1];
		            current_mapping[node_m1] = current_mapping[node_m2];
		            current_mapping[node_m2] = temp;
                            //current_score+=max_delta;
                            current_score = calculate_alignment_score(gm, gf, current_mapping);

		            improvements++;
            
		            if (current_score > best_score) {
		                memcpy(best_mapping, current_mapping, (max_node + 1) * sizeof(int));
		                best_score = current_score;
		                LOG_INFO("Process %d found new best score: %d in iter %d", rank, best_score,iter);
		                save_intermediate_mapping(out_path, best_mapping, max_node, gm, gf, best_score);
				last=iter;
		            }

                            num_changed=2;
			    if (node_m1<node_m2 ){
				    changed_vertices[0]=node_m1;
				    changed_vertices[1]=node_m2;
			    } else {
				    changed_vertices[1]=node_m1;
				    changed_vertices[0]=node_m2;
			    }

	                    fillArray(changed_vertices,num_changed,max_node);
			    // Upload changed vertices to GPU
			    cudaMemcpy(d_changed_vertices, changed_vertices, 
		               max_node * sizeof(int), cudaMemcpyHostToDevice);
    
	                    cudaMemcpy(d_mapping, current_mapping, (max_node + 1) * sizeof(int), cudaMemcpyHostToDevice);
			    // Calculate new number of pairs to check
		    	    total_pairs = num_changed * (num_changed-1)/2+ num_changed* (max_node-num_changed);
    
			    // Launch kernel with new parameters
			    threadsPerBlock = 256;
			    numBlocks = (total_pairs + threadsPerBlock - 1) / threadsPerBlock;
    
			    calculateSwapDeltasKernelQuick<<<numBlocks, threadsPerBlock>>>(
			        d_adj_matrix_m,
			        d_adj_matrix_f,
			        d_mapping,
			        max_node,
			        d_changed_vertices,
			        num_changed,
			        d_deltas,
			        d_node_pairs
			    );
		            cudaDeviceSynchronize();
		            cudaMemcpy(h_deltas, d_deltas, total_pairs * sizeof(int), cudaMemcpyDeviceToHost);
		            cudaMemcpy(h_node_pairs, d_node_pairs, total_pairs * 2 * sizeof(int), cudaMemcpyDeviceToHost);
	                    iter++;

			        // Find best improvement
			    max_delta = 0;
			    best_pair_idx = -1;
			    num_positives=0;
			    num_zeros=0;
			    for (int i = 0; i < total_pairs; i++) {
			            if (h_deltas[i] > max_delta) {
			                max_delta = h_deltas[i];
			                best_pair_idx = i;
			            }
				    if (h_deltas[i] > 0 ) {
			               positive_pairs[num_positives*2]=h_node_pairs[i*2];
			               positive_pairs[num_positives*2+1]=h_node_pairs[i*2+1];
				       num_positives++;
				    }
	 		            if (h_deltas[i] == 0 ) {
			               positive_pairs[num_pairs * 2-2-num_zeros*2]=h_node_pairs[i*2];
			               positive_pairs[num_pairs * 2-2-num_zeros*2+1]=h_node_pairs[i*2+1];
				       num_zeros++;
				    }
			    }
        		    current_time = time(NULL);
	                    if (iter % 2400 ==0) { 
    			            LOG_INFO("Average time for once quick brute-force search is  %f s for iteration %d, running time =%d s", (current_time-start_time)/(iter*1.0),iter,current_time-start_time);
			            LOG_INFO("Current score is %d  best score is %d of rank %d, improvement =%d", current_score, best_score, rank,improvements);
			    }

		    }



        } 

    }
    
    // Cleanup
    cudaFree(d_adj_matrix_m);
    cudaFree(d_adj_matrix_f);
    cudaFree(d_mapping);
    cudaFree(d_deltas);
    cudaFree(d_node_pairs);
    cudaFree(d_changed_vertices);

    free(h_deltas);
    free(h_node_pairs);
    free(positive_pairs);
    free(h_adj_matrix_m);
    free(h_adj_matrix_f);
    free(current_mapping);
    free(old_mapping);
    free(changed_vertices); 
    return best_mapping;
}

// Function to get maximum node ID from graph
int get_max_node(Graph* g) {
    return NUM_NODES;
}

// Function to clean up graph memory
void free_graph(Graph* g) {
    if (g->edges) {
        for (int i = 0; i < MAX_NODES; i++) {
            if (g->edges[i].count > 0) {
                free(g->edges[i].to_nodes);
                free(g->edges[i].weights);
            }
        }
        free(g->edges);
    }
    
    if (g->reverse_edges) {
        for (int i = 0; i < MAX_NODES; i++) {
            if (g->reverse_edges[i].count > 0) {
                free(g->reverse_edges[i].to_nodes);
                free(g->reverse_edges[i].weights);
            }
        }
        free(g->reverse_edges);
    }
    if (g->adj_matrix){
        for (int i=0; i<=NUM_NODES; i++){
            free(g->adj_matrix[i]);
        }
        free(g->adj_matrix);
        g->adj_matrix=NULL;
    }
    //free(g->nodes);
    free(g);
}

Graph* load_graph_from_csv(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filename);
        return NULL;
    }
    
    Graph* graph = new_graph();
    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    int total_lines = 0;
    
    // Count total lines for progress bar
    while (fgets(line, MAX_LINE_LENGTH, file)) total_lines++;
    rewind(file);
    
    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    total_lines--; // Adjust for header
    
    LOG_INFO("Loading graph from %s (%s lines)", filename, format_number(total_lines));
    
    time_t start_time = time(NULL);
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        int from, to, weight;
        if (sscanf(line, "%d,%d,%d", &from, &to, &weight) == 3) {
            add_edge(graph, from, to, weight);
            line_count++;
            if (line_count % 100000 == 0) {
                print_progress(line_count, total_lines, "Loading graph");
            }
        } else {
            LOG_ERROR("Malformed line in CSV: %s", line);
        }
    }
    
    time_t end_time = time(NULL);
    LOG_INFO("Graph loaded successfully:");
    LOG_INFO("  - Nodes: %s", format_number(NUM_NODES));
    LOG_INFO("  - Edges: %s", format_number(line_count));
    LOG_INFO("  - Time taken: %ld seconds", end_time - start_time);
    
    fclose(file);
    return graph;
}

// Main function
int main(int argc, char* argv[]) {



    // Initialize CUDA first
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("cudaSetDevice failed! Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print device info
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n", prop.name);











    int rank=0;
    int size=4;

    if (argc < 5) {
        if (rank == 0) {
            LOG_ERROR("Usage: %s <male graph> <female graph> <in mapping> <out mapping>", argv[0]);
        }
        return 1;
    }
	

    time_t total_start = time(NULL);
    // Only rank 0 prints initial information
    if (rank == 0) {
        LOG_INFO("Graph Alignment Tool v1.0 (MPI + OpenMP)");
        LOG_INFO("Number of MPI processes: %d", size);
        LOG_INFO("Starting process with:");
        LOG_INFO("  - Male graph: %s", argv[1]);
        LOG_INFO("  - Female graph: %s", argv[2]);
        LOG_INFO("  - Output mapping: %s", argv[4]);
    }

    
    Graph* gm = load_graph_from_csv(argv[1]);
    if (!gm) {
        LOG_ERROR("Failed to load male graph");
        return 1;
    }
    
    Graph* gf = load_graph_from_csv(argv[2]);
    if (!gf) {
        LOG_ERROR("Failed to load female graph");
        free_graph(gm);
        return 1;
    }
    
    int max_node = MAX(get_max_node(gm), get_max_node(gf));
    
    int* benchmark = load_benchmark_mapping(argv[3], max_node);
    if (!benchmark) {
        LOG_ERROR("Failed to load benchmark mapping");
        free_graph(gm);
        free_graph(gf);
        return 1;
    }
    
    int initial_score = calculate_alignment_score(gm, gf, benchmark);
    LOG_INFO("Initial alignment score: %s", format_number(initial_score));
    
    char outputfilename[100];

    sprintf(outputfilename, "ID%d-%s", rank,argv[4]);

    sprintf(outputfilename, "%.*s-rank%d.csv", (int)strlen(argv[4])-4, argv[4], rank);
    LOG_INFO("output file name is %s",outputfilename);

    int* optimized_mapping = optimize_mapping(gm, gf, benchmark, outputfilename);
    int optimized_score = calculate_alignment_score(gm, gf, optimized_mapping);
    
    time_t total_end = time(NULL);
    LOG_INFO("Process completed:");
    LOG_INFO("  - Initial score: %s", format_number(initial_score));
    LOG_INFO("  - Final score: %s", format_number(optimized_score));
    LOG_INFO("  - Improvement: %.2f%%",
            (double)(optimized_score - initial_score) / initial_score * 100.0);
    LOG_INFO("  - Total time: %.1f minutes", difftime(total_end, total_start) / 60);
    
    free_graph(gm);
    free_graph(gf);
    free(benchmark);
    free(optimized_mapping);
    
    return 0;
}
