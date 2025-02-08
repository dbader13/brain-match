# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall
MPICC = mpicc

#Directories
BEST_SWAP_DIR = best-swap/

# Directories for LAPACK/BLAS (update these if needed)
LAPACK_INCLUDE_DIR ?=  /scratch/users/zhihui/lapack-3.12.0/build/include/
LAPACK_LIB_DIR ?= /scratch/users/zhihui/lapack-3.12.0/build/lib/

INCLUDE_FLAGS ?=
LDFLAGS =  -lm

# Linker flags for LAPACK, BLAS, and math libraries
LAPACK_LDFLAGS = -L$(LAPACK_LIB_DIR) -llapack -lblas -lgfortran $(LDFLAGS)
LAPACK_INCLUDE_FLAGS = $(INCLUDE_FLAGS) -I$(LAPACK_INCLUDE_DIR)

# Targets for each executable
TARGETS = analyze-graph verify-mapping transform-male-female-VNC-inputs transform-solution convert-VNC-matching \
	score-matching identity-match spectral-match greedy-match greedy-feature-match \
	refine-matching-hillclimb refine-twohop graph-alignment graph-alignment-optimized graph-alignment-optimized-parallel \
	graph-alignment-permset-parallel \
	graph-re-alignment-mpi-omp \
	compute-betweenness-order re-alignment-parallel \
	score-lookup create-score-lookup parallel-create-score-lookup

# Source files for each program
ANALYZE_GRAPH_SRCS  = analyze-graph.c
VERIFY_MAPPING_SRCS = verify-mapping.c
TRANSFORM_SRCS      = transform-male-female-VNC-inputs.c
TRANSFORM_SOLN_SRCS = transform-solution.c
CONVERT_VNC_SRCS    = convert-VNC-matching.c
SCORE_MATCHING_SRCS = score-matching.c
IDENTITY_MATCH_SRCS = identity-match.c
SPECTRAL_MATCH_SRCS = spectral-match.c
GREEDY_MATCH_SRCS   = greedy-match.c
GREEDY_FEATURE_MATCH_SRCS   = greedy-feature-match.c
REFINE_MATCH_HILLCLIMB_SRCS = refine-matching-hillclimb.c
REFINE_TWOHOP_SRCS  = refine-twohop.c
GRAPH_ALIGN_SRCS    = graph-alignment.c
GRAPH_ALIGN_OPT_SRCS    = graph-alignment-optimized.c
GRAPH_ALIGN_OPT_PAR_SRCS    = graph-alignment-optimized-parallel.c
GRAPH_ALIGN_PERMSET_PAR_SRCS    = graph-alignment-permset-parallel.c
COMPUTE_BETWEENNESS_SRCS    = compute-betweenness-order.c
GRAPH_REALIGN_PAR_SRCS    = re-alignment-parallel.c
GRAPH_MPI_PAR_SRCS    = graph-re-alignment-mpi-omp.c
SCORE_LOOKUP_SRCS = $(BEST_SWAP_DIR)xoshiro.c $(BEST_SWAP_DIR)heap.c $(BEST_SWAP_DIR)utils.c $(BEST_SWAP_DIR)interrupts.c $(BEST_SWAP_DIR)parallel_hungarian.c $(BEST_SWAP_DIR)parallel_hungarian_subset.c $(BEST_SWAP_DIR)score-lookup.c 
CREATE_SCORE_LOOKUP_SRCS = $(BEST_SWAP_DIR)create-score-lookup.c
PARALLEL_CREATE_SCORE_LOOKUP_SRCS = $(BEST_SWAP_DIR)parallel_create_score_lookup.c


# Default target to build all programs
all: $(TARGETS)

# Analyze a connectome graph
analyze-graph: $(ANALYZE_GRAPH_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^ $(LDFLAGS)

# Verify a mapping file
verify-mapping: $(VERIFY_MAPPING_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^ $(LDFLAGS)

# Transform VNC male and female input graphs and matching to gm and gf
transform-male-female-VNC-inputs: $(TRANSFORM_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Transform a solution into VNC male and female matching
transform-solution: $(TRANSFORM_SOLN_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Convert the Codex Flywire VNC male and female matching to a generic one
convert-VNC-matching: $(CONVERT_VNC_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Build score-matching
score-matching: $(SCORE_MATCHING_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Build identity-match
identity-match: $(IDENTITY_MATCH_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Build spectral-match with LAPACK and BLAS
spectral-match: $(SPECTRAL_MATCH_SRCS)
	$(CC) $(CFLAGS) $(LAPACK_INCLUDE_FLAGS) -o $@ $^ $(LAPACK_LDFLAGS)

# Build Greedy match
greedy-match: $(GREEDY_MATCH_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Build Greedy feature match
greedy-feature-match: $(GREEDY_FEATURE_MATCH_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^ $(LDFLAGS)

# Build refine-matching-hillclimb
refine-matching-hillclimb: $(REFINE_MATCH_HILLCLIMB_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -pthread -o $@ $^ $(LDFLAGS)

# Build refine-twohop
refine-twohop: $(REFINE_TWOHOP_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^ $(LDFLAGS)

# Build graph-alignment
graph-alignment: $(GRAPH_ALIGN_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^ $(LDFLAGS)

# Build graph-alignment-optimized
graph-alignment-optimized: $(GRAPH_ALIGN_OPT_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^ $(LDFLAGS)

# Build graph-alignment-optimized-parallel
graph-alignment-optimized-parallel: $(GRAPH_ALIGN_OPT_PAR_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

# Build graph-alignment-permset-parallel
graph-alignment-permset-parallel: $(GRAPH_ALIGN_PERMSET_PAR_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

# Build compute-betweenness-order
compute-betweenness-order: $(COMPUTE_BETWEENNESS_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

# Build re-alignment-parallel
re-alignment-parallel: $(GRAPH_REALIGN_PAR_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

# Build re-alignment-parallel
graph-re-alignment-mpi-omp: $(GRAPH_MPI_PAR_SRCS)
	$(MPICC) $(CFLAGS) $(INCLUDE_FLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

score-lookup: $(SCORE_LOOKUP_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

create-score-lookup: $(CREATE_SCORE_LOOKUP_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^ $(LDFLAGS)

parallel-create-score-lookup: $(PARALLEL_CREATE_SCORE_LOOKUP_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

# Clean up build files
clean:
	rm -f $(TARGETS) *.o

# Phony targets to prevent conflicts with files named 'all' or 'clean'
.PHONY: all clean
