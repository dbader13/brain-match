#pragma once

#include <stdbool.h>
#include <stdint.h>

// Constants
#define MAX_LINE_LENGTH 1024
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)
#define MAX_NODES 100000
#define NUM_NODES 18524

// Logging definitions
typedef enum {
    LOG_LEVEL_INFO = 0,
    LOG_LEVEL_DEBUG = 1,
    LOG_LEVEL_ERROR = 2
} LogLevel;

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
    int *to_nodes;
    int *weights;
    int count;
    int capacity;
} EdgeMap;

typedef struct Graph {
    EdgeMap *edges;
    EdgeMap *reverse_edges;
    short *adj_matrix;
    int node_capacity;
} Graph;

// Function declarations
void print_progress(int current, int total, const char *prefix);
char *format_number(int num);
EdgeMap *new_edge_map(void);
Graph *new_graph(void);
void add_to_edge_map(EdgeMap *em, int to, int weight);
void add_edge(Graph *g, int from, int to, int weight);
int calculate_alignment_score(Graph *gm, Graph *gf, int *mapping);
int calculate_swap_delta(Graph *gm, Graph *gf, int *mapping, int node_m1, int node_m2);
int *load_mapping(const char *filename, int max_node);
void save_mapping(const char *filename, int *mapping);
int get_max_node(Graph *g);
void free_graph(Graph *g);
Graph *load_graph_from_csv(const char *filename);