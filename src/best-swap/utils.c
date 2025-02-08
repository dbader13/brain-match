#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>

#include "utils.h"

#define MAX_LINE_LENGTH 1024
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)
#define MAX_NODES 100000

#define CURRENT_LOG_LEVEL LOG_LEVEL_INFO

// Progress bar function
void print_progress(int current, int total, const char *prefix) {
    const int bar_width = 50;
    float progress = (float)current / total;
    int filled = (int)(bar_width * progress);

    printf("\r%s [", prefix);
    for (int i = 0; i < bar_width; i++) {
    if (i < filled)
        printf("=");
    else if (i == filled)
        printf(">");
    else
        printf(" ");
    }
    printf("] %.1f%%", progress * 100);
    fflush(stdout);
    if (current == total) printf("\n");
}

// Function to format numbers with commas
char *format_number(int num) {
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
    for (i = 0; i < j / 2; i++) {
        char t = formatted[i];
        formatted[i] = formatted[j - 1 - i];
        formatted[j - 1 - i] = t;
    }

    return formatted;
}

EdgeMap *new_edge_map() {
    EdgeMap *em = malloc(sizeof(EdgeMap));
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

Graph *new_graph() {
    Graph *g = malloc(sizeof(Graph));
    if (!g) {
    LOG_ERROR("Failed to allocate Graph");
    exit(1);
    }
    g->edges = NULL;
    g->reverse_edges = NULL;
    g->adj_matrix = NULL;
    g->adj_matrix =
        (short *)calloc((NUM_NODES + 1) * (NUM_NODES + 1), sizeof(short));
    g->node_capacity = 10000;
    return g;
}

void add_to_edge_map(EdgeMap *em, int to, int weight) {
    if (em->count >= em->capacity) {
        em->capacity *= 2;
        int *new_to_nodes = realloc(em->to_nodes, sizeof(int) * em->capacity);
        int *new_weights = realloc(em->weights, sizeof(int) * em->capacity);
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

void add_edge(Graph *g, int from, int to, int weight) {
    //    LOG_DEBUG("Adding edge: %d -> %d (weight: %d)", from, to, weight);
    g->adj_matrix[from * NUM_NODES + to] = (short) weight;

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

int calculate_alignment_score(Graph *gm, Graph *gf, int *mapping) {
    int score = 0;

    for (int src_m = 1; src_m <= NUM_NODES; src_m++) {
    // int src_m = gm->i;
        for (int j = 0; j < gm->edges[src_m].count; j++) {
            int dst_m = gm->edges[src_m].to_nodes[j];
            int weight_m = gm->edges[src_m].weights[j];
            int src_f = mapping[src_m];
            int dst_f = mapping[dst_m];
            score += MIN(weight_m, gf->adj_matrix[src_f * NUM_NODES + dst_f]);
        }
    }

    return score;
}

int calculate_swap_delta(Graph *gm, Graph *gf, int *mapping, int node_m1,
                         int node_m2) {
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
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f1 * NUM_NODES + dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f2 * NUM_NODES + dst_f]);

        delta += new_weight - old_weight;
    }

    // Handle incoming edges to node_m1
    for (int i = 0; i < gm->reverse_edges[node_m1].count; i++) {
        int src_m = gm->reverse_edges[node_m1].to_nodes[i];
        if (src_m == node_m2) continue;  // Skip direct edge between swapped nodes

        int weight_m = gm->reverse_edges[node_m1].weights[i];
        int src_f = mapping[src_m];

        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f * NUM_NODES + node_f1]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f * NUM_NODES + node_f2]);

        delta += new_weight - old_weight;
    }

    // Handle outgoing edges from node_m2
    for (int i = 0; i < gm->edges[node_m2].count; i++) {
        int dst_m = gm->edges[node_m2].to_nodes[i];
        if (dst_m == node_m1) continue;  // Skip direct edge between swapped nodes

        int weight_m = gm->edges[node_m2].weights[i];
        int dst_f = mapping[dst_m];

        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[node_f2 * NUM_NODES + dst_f]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[node_f1 * NUM_NODES + dst_f]);

        delta += new_weight - old_weight;
    }

    // Handle incoming edges to node_m2
    for (int i = 0; i < gm->reverse_edges[node_m2].count; i++) {
        int src_m = gm->reverse_edges[node_m2].to_nodes[i];
        if (src_m == node_m1) continue;  // Skip direct edge between swapped nodes

        int weight_m = gm->reverse_edges[node_m2].weights[i];
        int src_f = mapping[src_m];

        // Remove old contribution
        int old_weight = MIN(weight_m, gf->adj_matrix[src_f * NUM_NODES + node_f2]);
        // Add new contribution
        int new_weight = MIN(weight_m, gf->adj_matrix[src_f * NUM_NODES + node_f1]);

        delta += new_weight - old_weight;
    }

    // Handle direct edges between the swapped nodes
    // From m1 to m2
    short m1_to_m2 = gm->adj_matrix[node_m1 * NUM_NODES + node_m2];
    if (m1_to_m2 > 0) {
        int old_weight = MIN(m1_to_m2, gf->adj_matrix[node_f1 * NUM_NODES + node_f2]);
        int new_weight = MIN(m1_to_m2, gf->adj_matrix[node_f2 * NUM_NODES + node_f1]);
        delta += new_weight - old_weight;
    }

    // From m2 to m1
    short m2_to_m1 = gm->adj_matrix[node_m2 * NUM_NODES + node_m1];
    if (m2_to_m1 > 0) {
        int old_weight = MIN(m2_to_m1, gf->adj_matrix[node_f2 * NUM_NODES + node_f1]);
        int new_weight = MIN(m2_to_m1, gf->adj_matrix[node_f1 * NUM_NODES + node_f2]);
        delta += new_weight - old_weight;
    }

    return delta;
}

// Function to load benchmark mapping from CSV
int *load_mapping(const char *filename, int max_node) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filename);
        return NULL;
    }

    int *mapping = calloc(max_node + 1, sizeof(int));
    char line[MAX_LINE_LENGTH];
    int count = 0;

    // Try to get expected score from filename
    int expected_score = 0;
    const char *underscore = strrchr(filename, '_');
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

void save_mapping(const char *filename, int *mapping) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        LOG_ERROR("Error creating file: %s", filename);
        fclose(file);
        return;
    }

    fprintf(file, "Male Node ID,Female Node ID\n");
    for (int i = 1; i <= NUM_NODES; i++) {
        fprintf(file, "m%d,f%d\n", i, mapping[i]);
    }

    fclose(file);
}

int get_max_node(Graph *g) { return NUM_NODES; }

// Function to clean up graph memory
void free_graph(Graph *g) {
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

    if (g->adj_matrix) {
        free(g->adj_matrix);
        g->adj_matrix = NULL;
    }
    free(g);
}

Graph *load_graph_from_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
    LOG_ERROR("Failed to open file: %s", filename);
    return NULL;
    }

    Graph *graph = new_graph();
    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    int total_lines = 0;

    // Count total lines for progress bar
    while (fgets(line, MAX_LINE_LENGTH, file)) total_lines++;
    rewind(file);

    // Skip header
    fgets(line, MAX_LINE_LENGTH, file);
    total_lines--;  // Adjust for header

    LOG_INFO("Loading graph from %s (%s lines)", filename,
            format_number(total_lines));

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
