#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>

// Add at the beginning of the file, after other includes
#include <mpi.h>

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


// Modify main function and add sync function
void sync_best_solutions(int* best_mapping, int* best_score, int max_node) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Gather all scores at rank 0
    int* all_scores = NULL;
    if (rank == 0) {
        all_scores = (int*)malloc(size * sizeof(int));
    }

    
    
    MPI_Gather(best_score, 1, MPI_INT, all_scores, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Rank 0 finds the best score and its process
    int global_best_score = *best_score;
    int best_rank = rank;
    
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            if (all_scores[i] > global_best_score) {
                global_best_score = all_scores[i];
                best_rank = i;
            }
        }
        free(all_scores);
    }
    
    // Broadcast the rank with best score
    MPI_Bcast(&best_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_best_score, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == best_rank) {
            // Send best mapping to all other processes
            for (int i = 0; i < size; i++) {
                if (i != rank) {
                    MPI_Send(best_mapping, max_node + 1, MPI_INT, i, TAG_MAPPING, MPI_COMM_WORLD);
                }
            }
    } else {
            // Receive best mapping from the best process
            MPI_Recv(best_mapping, max_node + 1, MPI_INT, best_rank, TAG_MAPPING, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    *best_score = global_best_score;
    
    MPI_Barrier(MPI_COMM_WORLD);
}
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
    EdgeMap* em = malloc(sizeof(EdgeMap));
    if (!em) {
        LOG_ERROR("Failed to allocate EdgeMap");
        exit(1);
    }
    em->capacity = 100;
    em->count = 0;
    em->to_nodes = malloc(sizeof(int) * em->capacity);
    em->weights = malloc(sizeof(int) * em->capacity);
    if (!em->to_nodes || !em->weights) {
        LOG_ERROR("Failed to allocate EdgeMap arrays");
        exit(1);
    }
    return em;
}

Graph* new_graph() {
    Graph* g = malloc(sizeof(Graph));
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
        int* new_to_nodes = realloc(em->to_nodes, sizeof(int) * em->capacity);
        int* new_weights = realloc(em->weights, sizeof(int) * em->capacity);
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
        g->edges = calloc(MAX_NODES, sizeof(EdgeMap));
        g->reverse_edges = calloc(MAX_NODES, sizeof(EdgeMap));
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

    int* ordering = calloc(max_node + 1, sizeof(int));
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
    
    int* ordering = read_ordering(ordering_path, max_node);
    NodeMetrics* metrics = calloc(max_node + 1, sizeof(NodeMetrics));
    
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
    
    int* mapping = calloc(max_node + 1, sizeof(int));
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



// Function to generate a random number following a skewed normal distribution
int skewed_random(int n, double skew_factor) {
    double u = (double)rand() / RAND_MAX;
    double v = (double)rand() / RAND_MAX;
    double normal = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
    
    // Apply skewness
    double skewed = normal + skew_factor;
    
    // Scale to the desired range (80-90% of n)
    int range_start = (int)(0.8 * n);
    int range_end = (int)(0.9 * n);
    int range_size = range_end - range_start;
    
    // Map the skewed value to the 80-90% range
    int idx = range_start + (int)((skewed + 3.0) / 6.0 * range_size);
    
    // Ensure the index is within bounds
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    
    return idx;
}

void random_swap_k_vertices(int* mapping, int n, int k, int seed) {
    if (k*2 > n) {
        LOG_ERROR("K is too big : %d", k);
        return;
    }

    // Initialize random seed
    srand(time(NULL) + seed);

    // Create arrays for tracking
    int* selected = (int*)malloc(2*k * sizeof(int));
    int* used = (int*)calloc(n, sizeof(int));

    // Parameters for skewed normal distribution
    double mean = 0.85 * n;  // Center at 85% (middle of 80-90% interval)
    double std_dev = 0.05 * n;  // Standard deviation to cover 80-90% interval

    // Select vertices with skewed distribution
    int count = 0;
    while (count < 2*k) {
        // Generate normally distributed random number
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;

        // Box-Muller transform for normal distribution
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

        // Transform to our desired range
        int idx = (int)(mean + z * std_dev);

        // Ensure index is within bounds and not used
        if (idx >= 0 && idx < n && !used[idx]) {
            selected[count] = idx;
            used[idx] = 1;
            count++;
        }
    }

    // Store and swap values
    int* original_values = (int*)malloc(2*k * sizeof(int));
    for (int i = 0; i < 2*k; i++) {
        original_values[i] = mapping[selected[i]];
    }

    for (int i = 0; i < k; i++) {
        mapping[selected[i]] = original_values[i+k];
        mapping[selected[i+k]] = original_values[i];
    }

    // Clean up
    free(selected);
    free(used);
    free(original_values);
}
/*
void random_swap_k_vertices(int* mapping, int n, int k, int seed) {
    // Create array for tracking selected vertices
    int* selected = (int*)malloc(2 * k * sizeof(int));
    int* used = (int*)calloc(n, sizeof(int));  // Track used positions

    if (k * 2 > n) {
        LOG_ERROR("K is too big : %d", k);
        free(selected);
        free(used);
        return;
    }

    // Randomly select k vertices with a skewed distribution
    int count = 0;
    srand(time(NULL) + seed);

    while (count < 2 * k) {
        int idx = skewed_random(n, 1.0);  // Adjust skew_factor as needed
        if (!used[idx]) {
            selected[count] = idx;
            used[idx] = 1;
            count++;
        }
    }

    // Store original values
    int* original_values = (int*)malloc(2 * k * sizeof(int));
    for (int i = 0; i < 2 * k; i++) {
        original_values[i] = mapping[selected[i]];
    }

    // Swap the values
    for (int i = 0; i < k; i++) {
        mapping[selected[i]] = original_values[i + k];
        mapping[selected[i + k]] = original_values[i];
    }

    // Clean up
    free(selected);
    free(used);
    free(original_values);
}




void random_swap_k_vertices(int* mapping, int n, int k,int seed) {
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
        int idx = rand() % n;
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

    for (int i = 0; i < k; i++) {
        mapping[selected[i]] = original_values[i+k];
        mapping[selected[i+k]] = original_values[i];
    }

    // Clean up
    free(selected);
    free(used);
    free(original_values);
}

*/

// Function to optimize mapping
int* optimize_mapping(Graph* gm, Graph* gf, int* initial_mapping,
                     const char* out_path) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int max_node = NUM_NODES;

    int* current_mapping = malloc(sizeof(int) * (max_node + 1));
    int* best_mapping = malloc(sizeof(int) * (max_node + 1));
    int* old_mapping = malloc(sizeof(int) * (max_node + 1));
    memcpy(current_mapping, initial_mapping, sizeof(int) * (max_node + 1));
    memcpy(best_mapping, current_mapping, sizeof(int) * (max_node + 1));


    
    int current_score = calculate_alignment_score(gm, gf, current_mapping);
    int best_score=current_score;
    int outer_node_count = 0;
    int improvements = 0;
    time_t start_time = time(NULL);
    random_swap_k_vertices(current_mapping,  max_node,  10+rank,rank );
    current_score = calculate_alignment_score(gm, gf, current_mapping);

    int pass=0;
    int last=0;
    time_t last_sync_time = start_time;
    int iter=0;

    // Each process starts with a different random perturbation

    LOG_INFO("Process %d, Starting optimization with initial score: %s", rank, format_number(current_score));


    while (true){

	// Check if it's time to sync
	
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

              sync_best_solutions(best_mapping, &best_score, max_node);
              last_sync_time = current_time;
              LOG_INFO("Process %d, after synchronization and best score is %d", rank,best_score);

        }

    // Optimization loop

    for (int node_m1=1; node_m1<=NUM_NODES; node_m1++) {
        //int node_m1 = gm->nodes[i];
        outer_node_count++;
        
	if (outer_node_count % UPDATE_INTERVAL == 0 ) {
                time_t current_time = time(NULL);
                double elapsed = difftime(current_time, start_time);
                double nodes_per_sec = outer_node_count / elapsed;

                LOG_INFO("Optimization progress (Process %d):", rank);
                LOG_INFO("  - Current score: %s", format_number(current_score));
                LOG_INFO("  - Current best score: %s", format_number(best_score));
                LOG_INFO("  - Improvements found: %s", format_number(improvements));
        }

        int maxdelta=-1;
	int maxv=-1;
        #pragma omp parallel for shared (maxdelta,maxv) schedule(dynamic)
        for (int node_m2=node_m1+1; node_m2<=NUM_NODES; node_m2++) {
            if (node_m1 == node_m2) continue;
            
            int node_f1 = current_mapping[node_m1];
            int node_f2 = current_mapping[node_m2];
	    
            
            int delta = calculate_swap_delta(gm, gf, current_mapping, node_m1, node_m2);
                
            #pragma omp critical
	    {
                    if (delta > maxdelta) {
			maxdelta=delta;
			maxv=node_m2;
		    }
	    }
        }
	if (maxdelta>0) {
                    // Save current mapping for validation
                    memcpy(old_mapping, current_mapping, sizeof(int) * (max_node + 1));
		    
                    int node_f1 = current_mapping[node_m1];
                    int node_f2 = current_mapping[maxv];
                    
                    // Make the swap
                    current_mapping[node_m1] = node_f2;
                    current_mapping[maxv] = node_f1;
                    current_score += maxdelta;
                    improvements++;
		    if (current_score > best_score) {
                          memcpy(best_mapping, current_mapping, sizeof(int) * (max_node + 1));
			  best_score=current_score;
                          LOG_INFO("Current process ID is %d",rank);
                          LOG_INFO("Saving Best Matching");
                          LOG_INFO("  - Current score: %s", format_number(current_score));
                          LOG_INFO("  - Current best score: %s", format_number(best_score));
                          LOG_INFO("  - Write to mapping file")
                          save_intermediate_mapping(out_path, best_mapping, max_node, 
                                   gm, gf, best_score);
		    }
		    
                    // Validate changes
                    validate_mapping_changes(old_mapping, current_mapping, max_node, 
                                          node_m1, maxv);
                    
                    // Verify score calculation
                    int verify_score = calculate_alignment_score(gm, gf, current_mapping);
                    if (verify_score != current_score) {
                        LOG_ERROR("Score mismatch after swap - calculated: %d, verified: %d",
                                current_score, verify_score);
                        current_score = verify_score;  // Trust the verification
                    }
        }
    }
    if (last<improvements) {
	last=improvements;
	pass=0;
    } else {
        pass++;
    }
    if (pass >2 && improvements-last==0 ) { 
                time_t current_time = time(NULL);
                double elapsed = difftime(current_time, start_time);
		int shuffle=3;
		shuffle=shuffle+rand() % 3 ;
                LOG_INFO("Process %d Restart Optimization from a perturbation of the best configuration ",rank);
                LOG_INFO("  - Elapsed time is %f ",elapsed);
                LOG_INFO("  - Without Progress in %d passes ",pass);
                LOG_INFO("  - Randomly shuffle %d vertices",shuffle+rank);
                memcpy(current_mapping, best_mapping, sizeof(int) * (max_node + 1));
                random_swap_k_vertices(current_mapping,  max_node,  shuffle, rank);
                current_score = calculate_alignment_score(gm, gf, current_mapping);
		improvements=0;
		last=0;
		pass=0;
    } 
    current_time = time(NULL);
    iter++;
    if (iter % 2  == 0  && rank==0) {
                double elapsed = difftime(current_time, start_time);
                double average_time = elapsed/iter;

                LOG_INFO("Optimization progress %d iteration (Process %d):", iter,rank);
                LOG_INFO("  - Current score: %s", format_number(current_score));
                LOG_INFO("  - Current best score: %s", format_number(best_score));
                LOG_INFO("  - Improvements found: %s", format_number(improvements));
                LOG_INFO("  - Average time of once iteration : %f", average_time);
    }
    }// end of while loop
    
    memcpy(best_mapping, current_mapping, sizeof(int) * (max_node + 1));
    time_t end_time = time(NULL);
    LOG_INFO("Optimization completed:");
    LOG_INFO("  - Final score: %s", format_number(best_score));
    LOG_INFO("  - Total improvements: %s", format_number(improvements));
    LOG_INFO("  - Time taken: %.1f minutes", difftime(end_time, start_time) / 60);
    
    // Save final mapping with verification
    save_intermediate_mapping(out_path, best_mapping, max_node, gm, gf, best_score);
    
    free(current_mapping);
    free(old_mapping);
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 5) {
        if (rank == 0) {
            LOG_ERROR("Usage: %s <male graph> <female graph> <in mapping> <out mapping>", argv[0]);
        }
        MPI_Finalize();
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
    MPI_Finalize();
    
    return 0;
}
