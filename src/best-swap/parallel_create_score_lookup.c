#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <signal.h>
#include <omp.h>

#define SAVE_INTERVAL 25
#define UPDATE_INTERVAL 20
#define MAX_LINE_LENGTH 1024
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MAX_NODES 100000
#define NUM_NODES 18524

static volatile sig_atomic_t interrupted = 0;

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
    short* adj_matrix;
    int node_capacity;
} Graph;

void sigint_handler(int sig) {
    interrupted = 1;
}

// Function to format numbers with commas
// Modified to be thread-safe by removing static buffer
char* format_number(int num) {
    char* formatted = malloc(32);
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
    g->adj_matrix = (short*)calloc((NUM_NODES+1) * (NUM_NODES+1), sizeof(short));
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

void add_edge(Graph* g, int from, int to, int weight) {
  //    LOG_DEBUG("Adding edge: %d -> %d (weight: %d)", from, to, weight);
    g->adj_matrix[from*NUM_NODES + to] = (short) weight;
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

// Parallelized version of calculate_alignment_score
int calculate_alignment_score(Graph* gm, Graph* gf, int* mapping) {
    int total_score = 0;
    
    #pragma omp parallel reduction(+:total_score)
    {
        #pragma omp for schedule(dynamic, 100)
        for (int src_m = 1; src_m <= NUM_NODES; src_m++) {
            int local_score = 0;
            for (int j = 0; j < gm->edges[src_m].count; j++) {
                int dst_m = gm->edges[src_m].to_nodes[j];
                int weight_m = gm->edges[src_m].weights[j];
                int src_f = mapping[src_m];
                int dst_f = mapping[dst_m];
                local_score += MIN(weight_m, gf->adj_matrix[src_f*NUM_NODES + dst_f]);
            }
            total_score += local_score;
        }
    }
    
    return total_score;
}

// calculate_swap_delta remains unchanged as it's called within parallel regions
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
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f1*NUM_NODES + dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f2*NUM_NODES + dst_f]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle incoming edges to node_m1
    for (int i = 0; i < gm->reverse_edges[node_m1].count; i++) {
        int src_m = gm->reverse_edges[node_m1].to_nodes[i];
        if (src_m == node_m2) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->reverse_edges[node_m1].weights[i];
        int src_f = mapping[src_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f*NUM_NODES + node_f1]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f*NUM_NODES + node_f2]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle outgoing edges from node_m2
    for (int i = 0; i < gm->edges[node_m2].count; i++) {
        int dst_m = gm->edges[node_m2].to_nodes[i];
        if (dst_m == node_m1) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->edges[node_m2].weights[i];
        int dst_f = mapping[dst_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f2*NUM_NODES + dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f1*NUM_NODES + dst_f]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle incoming edges to node_m2
    for (int i = 0; i < gm->reverse_edges[node_m2].count; i++) {
        int src_m = gm->reverse_edges[node_m2].to_nodes[i];
        if (src_m == node_m1) continue;  // Skip direct edge between swapped nodes
        
        int weight_m = gm->reverse_edges[node_m2].weights[i];
        int src_f = mapping[src_m];
        
        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f*NUM_NODES + node_f2]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f*NUM_NODES + node_f1]);
        
        delta += new_weight - old_weight;
    }
    
    // Handle direct edges between the swapped nodes
    // From m1 to m2
    short m1_to_m2 = gm->adj_matrix[node_m1*NUM_NODES + node_m2];
    if (m1_to_m2 > 0) {
        int old_weight = MIN(m1_to_m2, gf->adj_matrix[node_f1*NUM_NODES + node_f2]);
        int new_weight = MIN(m1_to_m2, gf->adj_matrix[node_f2*NUM_NODES + node_f1]);
        delta += new_weight - old_weight;
    }
    
    // From m2 to m1
    short m2_to_m1 = gm->adj_matrix[node_m2*NUM_NODES + node_m1];
    if (m2_to_m1 > 0) {
        int old_weight = MIN(m2_to_m1, gf->adj_matrix[node_f2*NUM_NODES + node_f1]);
        int new_weight = MIN(m2_to_m1, gf->adj_matrix[node_f1*NUM_NODES + node_f2]);
        delta += new_weight - old_weight;
    }

    //Handle direct edges between the resultants (f1 and f2)
    // short f1_to_f2 = gf->adj_matrix[node_f1 * NUM_NODES + node_f2];
    // if (f1_to_f2 > 0) {
    //     int old_weight = MIN(f1_to_f2, gf->adj_matrix[node_f1 * NUM_NODES + node_f2]);
    //     int new_weight = MIN(f1_to_f2, gf->adj_matrix[node_f2 * NUM_NODES + node_f1]);
    //     delta += new_weight - old_weight;
    // }

    // short f2_to_f1 = gf->adj_matrix[node_f2 * NUM_NODES + node_f1];
    // if (f2_to_f1 > 0) {
    //     int old_weight = MIN(f2_to_f1, gf->adj_matrix[node_f2 * NUM_NODES + node_f1]);
    //     int new_weight = MIN(f2_to_f1, gf->adj_matrix[node_f1 * NUM_NODES + node_f2]);
    //     delta += new_weight - old_weight;
    // }
    
    return delta;
}

short* create_score_lookup(Graph* gm, Graph* gf, int* mapping) {
    LOG_INFO("Creating score lookup using OpenMP...");
    short* lookup = (short*)calloc(NUM_NODES * NUM_NODES, sizeof(short));
    
    const int maxCount = NUM_NODES * (NUM_NODES-1) / 2;
    volatile long progress = 0;  // Shared progress counter
    
    // Initialize diagonal elements first
    for (int i = 0; i < NUM_NODES; i++) {
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
            lookup[i*NUM_NODES + j] = delta;
            // lookup[j*NUM_NODES + i] = (short) calculate_swap_delta(gm, gf, mapping, j+1, i+1);
            lookup[j*NUM_NODES + i] = delta;

            
            #pragma omp atomic
            progress++;
            
            if ((progress & 0x3FFFF) == 0) {
                #pragma omp critical
                {
                    // LOG_INFO("Processed %ld swaps out of %d - %.2f%% complete", 
                    //         progress, maxCount, 
                    //         ((double)progress / (double)maxCount) * 100.0);
                    print_progress(progress, maxCount, "Creating score lookup");
                }
            }
        }
    }
    
    return lookup;
}
void save_lookup(const char* filename, short* lookup) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        LOG_ERROR("Error creating file: %s", filename);
        exit(1);
    }
    
    for (int i = 0; i < NUM_NODES; i++) {
        for (int j = 0; j < NUM_NODES; j++) {
            fprintf(file, "%d,", lookup[i*NUM_NODES + j]);
        }
    }
    
    fclose(file);
}

// Function to load benchmark mapping from CSV
int* load_mapping(const char* filename, int max_node) {
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
        free(g->adj_matrix);
        g->adj_matrix=NULL;
    }
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
    LOG_INFO("  - Time taken: %lld seconds", end_time - start_time);
    
    fclose(file);
    return graph;
}


// Modified main function to set up OpenMP
int main(int argc, char* argv[]) {
    if (argc < 5) {
        LOG_ERROR("Usage: %s <male graph> <female graph> <in mapping> <out lookup>", argv[0]);
        return 1;
    }

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);
    // signal(SIGUSR2, sigint_handler);
    
    // Set OpenMP parameters
    omp_set_nested(1);  // Enable nested parallelism
    
    // Get number of available threads
    int max_threads = omp_get_max_threads();
    LOG_INFO("OpenMP Score Lookup Tool v1.0");
    LOG_INFO("Available threads: %d", max_threads);
    
    time_t total_start = time(NULL);
    LOG_INFO("Starting process with:");
    LOG_INFO("  - Male graph: %s", argv[1]);
    LOG_INFO("  - Female graph: %s", argv[2]);
    LOG_INFO("  - Output mapping: %s", argv[4]);
    
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
    
    int* benchmark = load_mapping(argv[3], max_node);
    if (!benchmark) {
        LOG_ERROR("Failed to load benchmark mapping");
        free_graph(gm);
        free_graph(gf);
        return 1;
    }

    short* lookup = create_score_lookup(gm, gf, benchmark);
    save_lookup(argv[4], lookup);
    
    int initial_score = calculate_alignment_score(gm, gf, benchmark);
    char* score_str = format_number(initial_score);
    LOG_INFO("Initial alignment score: %s", score_str);
    free(score_str);
    
    time_t total_end = time(NULL);
    char* final_score_str = format_number(initial_score);
    LOG_INFO("Process completed:");
    LOG_INFO("  - Initial score: %s", final_score_str);
    LOG_INFO("  - Total time: %.1f minutes", difftime(total_end, total_start) / 60);
    free(final_score_str);
    
    free_graph(gm);
    free_graph(gf);
    free(lookup);
    free(benchmark);
    
    return 0;
}