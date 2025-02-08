#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>
#include "hungarian.h"


#define SAVE_INTERVAL 25
#define UPDATE_INTERVAL 20
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

int one_side_swap_delta(Graph* gm, Graph* gf, int* mapping, int node_m1, int node_m2) {
    int node_f1 = mapping[node_m1];
    int node_f2 = mapping[node_m2];
    int delta = 0;

    if (node_m1 == node_m2) {
        return 0;
    }

    // Handle outgoing edges from node_m1
    for (int i = 0; i < gm->edges[node_m1].count; i++) {
        int dst_m = gm->edges[node_m1].to_nodes[i];
        if (dst_m == node_m2 || dst_m == node_m1) {
            continue;  // Skip direct edge between swapped nodes
        }
        int weight_m = gm->edges[node_m1].weights[i];
        int dst_f = mapping[dst_m];

        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f1][dst_f]);
        delta -= old_weight;

        // Add new potential contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f2][dst_f]);
        delta += new_weight;
    }

    // Handle incoming edges to node_m1
    for (int i = 0; i < gm->reverse_edges[node_m1].count; i++) {
        int src_m = gm->reverse_edges[node_m1].to_nodes[i];
        if (src_m == node_m2 || src_m == node_m1) {
            continue;  // Skip direct edge between swapped nodes
        }
        int weight_m = gm->reverse_edges[node_m1].weights[i];
        int src_f = mapping[src_m];

        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f1]);
        delta -= old_weight;

        // Add new potential contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f][node_f2]);
        delta += new_weight;
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
    
    LOG_INFO("Loaded %s mappings", format_number(count));
    
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

int* optimal_assignment(int k, double** cost_matrix){
    hungarian_problem_t problem;
    int mode = HUNGARIAN_MODE_MAXIMIZE_UTIL; // Solve for maximizing cost
    int matrix_size = hungarian_init(&problem, cost_matrix, k, k, mode);

    // Solve the assignment problem
    hungarian_solve(&problem);

    int* assignment = return_assignment(&problem);
    hungarian_free(&problem);
    
    return assignment;
}

void swap(int *x, int *y) {
    int temp = *x;
    *x = *y;
    *y = temp;
}


// State for xoshiro256**
static uint64_t s[4];

// Seed the generator
void xoshiro_seed(uint64_t seed) {
    s[0] = seed;
    s[1] = 0x123456789ABCDEF;
    s[2] = 0xFEDCBA987654321;
    s[3] = 0xABCDEF012345678;
}

// Generate a 64-bit random number
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro_next(void) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}

// Generate a 32-bit random integer
int xoshiro_rand(void) {
    return (int)(xoshiro_next() >> 32);
}


// Function to optimize mapping
int* optimize_mapping(Graph* gm, Graph* gf, int* initial_mapping,
                     const char* male_ordering_path, const char* female_ordering_path,
                     const char* out_path) {
    int max_node = NUM_NODES;

    int* current_mapping = malloc(sizeof(int) * (max_node + 1));
    int* old_mapping = malloc(sizeof(int) * (max_node + 1));  // For validation

    memcpy(current_mapping, initial_mapping, sizeof(int) * (max_node + 1));
    int* rev_mapping = malloc(sizeof(int) * (max_node + 1));
    for (int i=1; i<=max_node; i++){
        rev_mapping[current_mapping[i]] = i;
    }
    NodeMetrics* metrics_m = calculate_node_metrics(gm, male_ordering_path);
    NodeMetrics* metrics_f = calculate_node_metrics(gf, female_ordering_path);
    
    int current_score = calculate_alignment_score(gm, gf, current_mapping);
    int outer_node_count = 0;
    int improvements = 0;
    time_t start_time = time(NULL);
    
    LOG_INFO("Starting optimization with initial score: %s", format_number(current_score));
    srand(time(NULL));
    int k=1000; //choose 5 independent nodes
    int iter=0;
    bool mode=false; //false means start with swaps
    xoshiro_seed(time(NULL));

    while (true){
        iter++;


        //random greedy swaps
        //if (mode==false) {
        //    int node1 = (rand()%NUM_NODES);
        //    int node2 = (rand()%NUM_NODES);
        //    int delta=calculate_swap_delta(gm,gf,current_mapping,node1, node2);
        //    if (true){
        //        //swap(current_mapping[node1],current_mapping[node2]);
        //        //swap(rev_mapping[node1],rev_mapping[node2]);
        //        int tmp=current_mapping[node1];
        //        current_mapping[node1]=current_mapping[node2];
        //        current_mapping[node2]=tmp;
        //        current_score+=delta;
        //    }
        //    if (iter%100000==0){
        //        printf("Iter: %i, Score: %i\n", iter, current_score);
        //    }
        //    if (iter%1000000==0)save_intermediate_mapping(out_path, current_mapping, max_node, gm, gf, current_score);
        //    continue;
        //}
        if (iter%100 == 0) mode=false;
        //hungarian optimization
        bool* available = malloc(sizeof(int) * (NUM_NODES + 1));
        int rem_nodes = NUM_NODES;
        for (int i=1; i<=NUM_NODES; i++){
            available[i]=true;
            //if (gm->adj_matrix[i][i] > 0 || gf->adj_matrix[current_mapping[i]][current_mapping[i]]>0){
            //    available[i]=false;
            //    rem_nodes--;
            //}
        }
        int* subset = malloc(sizeof(int) * (k));
        //printf("YES\n");

        for (int i=0; i<k; i++){
            //printf("YES\n");

            //printf("i: %i, rem_nodes: %i", i, rem_nodes);
            if (rem_nodes<=0) {
                k=i; 
                break;
            }
            int rand_ind = ((rand()) % rem_nodes)+1;
            int chosen=-1;
            for (int node=1; node<=NUM_NODES; node++){  
                if (available[node]==false) continue;
                rand_ind--;
                if (rand_ind == 0){
                    chosen = node;
                    break;
                }
            }
            if (chosen==-1){
                printf("NOT FOUND ERROR!\n");
            }

            rem_nodes--;
            available[chosen]=false;
            subset[i] = chosen;
            //now, update all neighbors
            for (int j = 0; j < gm->edges[chosen].count; j++) {
                int dst = gm->edges[chosen].to_nodes[j];
                if (available[dst]){
                    rem_nodes--;
                }
                available[dst]=false;
            }
            for (int j = 0; j < gm->reverse_edges[chosen].count; j++) {
                int dst = gm->reverse_edges[chosen].to_nodes[j];
                if (available[dst]){
                    rem_nodes--;
                }
                available[dst]=false;
            }
            //for (int j = 0; j < gf->edges[current_mapping[chosen]].count; j++) {
            //    int dst = gf->edges[current_mapping[chosen]].to_nodes[j];
            //    if (available[rev_mapping[dst]]){
            //        rem_nodes--;
            //    }
            //    available[rev_mapping[dst]]=false;
            //}
            //for (int j = 0; j < gf->reverse_edges[current_mapping[chosen]].count; j++) {
            //    int dst = gf->reverse_edges[current_mapping[chosen]].to_nodes[j];
            //    if (available[rev_mapping[dst]]){
            //        rem_nodes--;
            //    }
            //    available[rev_mapping[dst]]=false;
            //}
            //for (int j=1; j<=NUM_NODES; j++){
            //    if (available[rev_mapping[j]]==true && (gf->adj_matrix[current_mapping[chosen]][j] > 0 || gf->adj_matrix[j][current_mapping[chosen]] > 0)){
            //        printf("INCONSISTENT\n");
            //        //available[rev_mapping[j]]=false;
            //        //available[j]=false;
            //    }
            //}

        }
        //for (int i=0; i<k; i++){
        //    for (int j=0; j<k; j++) if (i!=j){
        //        if (gm->adj_matrix[subset[i]][subset[j]]>0 || gf->adj_matrix[current_mapping[subset[i]]][current_mapping[subset[j]]]>0){
        //            printf("CONNECTION\n");
        //        }
        //    }
        //}
        //printf("YES\n");
        free(available);

        //bool ok=true;
        //for (int i=0; i<k; i++){
        //    for (int j=0; j<k; j++) if (i!=j){
        //        if (gm->adj_matrix[subset[i]][subset[j]] > 0)
        //    }
        //}

        if (rem_nodes>=NUM_NODES-k){
            printf("NOT ABLE TO FIND LARGE ENOUGH, %i\n", rem_nodes);
        }
        int* prevMapping = malloc(sizeof(int) * (k));
        for (int i=0; i<k; i++){
            prevMapping[i] = current_mapping[subset[i]];
            //printf("%i: %i\n", i,subset[i]);
        }

        //printf("STEP 4\n");
        double** cost_matrix = (double**)calloc(k, sizeof(double*));
        for (int i = 0; i < k; i++) {
            cost_matrix[i] = (double*)calloc(k, sizeof(double));
        }
        //printf("k: %i\n", k);
        #pragma omp parallel for
        for (int i=0; i<k; i++){
            for (int j=0; j<k; j++){
                if (i==j) {
                    cost_matrix[i][j] = 0;
                    continue;
                }
                int m_i = subset[i];
                int m_j = subset[j]; 
                //cost_matrix[i][j] = 0;
                cost_matrix[i][j] = one_side_swap_delta(gm, gf, current_mapping, m_i, m_j);
                //printf("%f ", cost_matrix[i][j]);
            }
            //printf("\n");
        }

        int* assignments = optimal_assignment(k, cost_matrix);
        int delta = 0;
        int changed=0;
        for (int i=0; i<k; i++){
            //printf("%i, %i\n", i, assignments[i]);
            if (assignments[i]!=i){
                changed++;
            }
            delta += cost_matrix[i][assignments[i]];
            current_mapping[subset[i]] = prevMapping[assignments[i]];
            rev_mapping[prevMapping[assignments[i]]] = subset[i];
        }

        current_score += delta;
        current_score = calculate_alignment_score(gm, gf, current_mapping);

        printf("Iter: %i, Score: %i, Delta: %i, Changed: %i, K: %i\n", iter, current_score, delta, changed, k);
        for (int i = 0; i < k; i++) {
            free(cost_matrix[i]);
        }
        free(cost_matrix);
        free(assignments);
        free(prevMapping);
        free(subset);


        if (iter%5==0)save_intermediate_mapping(out_path, current_mapping, max_node, gm, gf, current_score);

        

        //while (true){
        //}
    }

    //while (true){
    //// Optimization loop
    //for (int node_m1=1; node_m1<=NUM_NODES; node_m1++) {
    //    //int node_m1 = gm->nodes[i];
    //    outer_node_count++;
    //    
    //    if (outer_node_count % UPDATE_INTERVAL == 0) {
    //        time_t current_time = time(NULL);
    //        double elapsed = difftime(current_time, start_time);
    //        double nodes_per_sec = outer_node_count / elapsed;
    //        
    //        LOG_INFO("Optimization progress:");
    //        LOG_INFO("  - Processed: %s/%s nodes (%.1f%%)",
    //                format_number(outer_node_count),
    //                format_number(NUM_NODES),
    //                (double)outer_node_count/NUM_NODES * 100);
    //        LOG_INFO("  - Current score: %s", format_number(current_score));
    //        LOG_INFO("  - Improvements found: %s", format_number(improvements));
    //        LOG_INFO("  - Processing speed: %.1f nodes/sec", nodes_per_sec);
    //        LOG_INFO("  - Estimated time remaining: %.1f minutes",
    //                (NUM_NODES - outer_node_count) / nodes_per_sec / 60);
    //    }
    //    
    //    if (outer_node_count % SAVE_INTERVAL == 0) {
    //        LOG_DEBUG("Saving intermediate mapping to %s", out_path);
    //        save_intermediate_mapping(out_path, current_mapping, max_node, 
    //                               gm, gf, current_score);
    //    }
    //    
    //    for (int node_m2=1; node_m2<=NUM_NODES; node_m2++) {
    //        //int node_m2 = gm->nodes[j];
    //        if (node_m1 == node_m2) continue;
    //        
    //        int node_f1 = current_mapping[node_m1];
    //        int node_f2 = current_mapping[node_m2];
    //        
    //        double current_sim = calculate_node_similarity(metrics_m[node_m1], metrics_f[node_f1]) +
    //                           calculate_node_similarity(metrics_m[node_m2], metrics_f[node_f2]);
    //        
    //        double swapped_sim = calculate_node_similarity(metrics_m[node_m1], metrics_f[node_f2]) +
    //                           calculate_node_similarity(metrics_m[node_m2], metrics_f[node_f1]);
    //        
    //        if (true) {
    //            int delta = calculate_swap_delta(gm, gf, current_mapping, node_m1, node_m2);
    //            
    //            if (delta > 0) {
    //                // Save current mapping for validation
    //                memcpy(old_mapping, current_mapping, sizeof(int) * (max_node + 1));
    //                
    //                // Make the swap
    //                current_mapping[node_m1] = node_f2;
    //                current_mapping[node_m2] = node_f1;
    //                current_score += delta;
    //                improvements++;
    //                
    //                // Validate changes
    //                validate_mapping_changes(old_mapping, current_mapping, max_node, 
    //                                      node_m1, node_m2);
    //                
    //                // Verify score calculation
    //                int verify_score = calculate_alignment_score(gm, gf, current_mapping);
    //                if (verify_score != current_score) {
    //                    LOG_ERROR("Score mismatch after swap - calculated: %d, verified: %d",
    //                            current_score, verify_score);
    //                    current_score = verify_score;  // Trust the verification
    //                }
    //            }
    //        }
    //    }
    //}
    //}
    
    time_t end_time = time(NULL);
    LOG_INFO("Optimization completed:");
    LOG_INFO("  - Final score: %s", format_number(current_score));
    LOG_INFO("  - Total improvements: %s", format_number(improvements));
    LOG_INFO("  - Time taken: %.1f minutes", difftime(end_time, start_time) / 60);
    
    // Save final mapping with verification
    save_intermediate_mapping(out_path, current_mapping, max_node, gm, gf, current_score);
    
    free(metrics_m);
    free(metrics_f);
    free(old_mapping);
    return current_mapping;
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
    srand( (unsigned) time(NULL) * getpid());
    omp_set_num_threads(12);
    if (argc < 7) {
        LOG_ERROR("Usage: %s <male graph> <female graph> <male ordering> <female ordering> <in mapping> <out mapping>", argv[0]);
        return 1;
    }
    
    time_t total_start = time(NULL);
    LOG_INFO("Graph Alignment Tool v1.0");
    LOG_INFO("Starting process with:");
    LOG_INFO("  - Male graph: %s", argv[1]);
    LOG_INFO("  - Female graph: %s", argv[2]);
    LOG_INFO("  - Output mapping: %s", argv[6]);
    
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
    
    int* benchmark = load_benchmark_mapping(argv[5], max_node);
    if (!benchmark) {
        LOG_ERROR("Failed to load benchmark mapping");
        free_graph(gm);
        free_graph(gf);
        return 1;
    }
    
    int initial_score = calculate_alignment_score(gm, gf, benchmark);
    LOG_INFO("Initial alignment score: %s", format_number(initial_score));
    
    int* optimized_mapping = optimize_mapping(gm, gf, benchmark, argv[3], argv[4], argv[6]);
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
