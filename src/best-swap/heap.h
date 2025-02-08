#pragma once

#include <stdbool.h>

// Struct representing a Swap
typedef struct {
    int nodeM1; // First male node in the swap
    int nodeM2; // Second male node in the swap
    int delta;   // Delta value of the swap
} Swap;

// Struct representing a bounded max heap of Swaps
typedef struct {
    Swap *data;      // Array of Swaps
    int capacity;    // Maximum capacity of the heap
    int size;        // Current number of elements in the heap
} SwapHeap;

// Function prototypes

/**
 * Create a max heap with the given capacity.
 *
 * @param capacity The maximum number of elements the heap can hold.
 * @return Pointer to the newly created heap.
 */
SwapHeap *create_swap_heap(int capacity);

/**
 * Insert a new Swap into the heap. If the heap is full and the new Swap's
 * delta is greater than the smallest delta in the heap, it replaces the
 * smallest delta Swap.
 *
 * @param heap Pointer to the MaxHeap.
 * @param swap The Swap to insert.
 */
void insert_swap(SwapHeap *heap, Swap swap);

/**
 * Extract the Swap with the highest delta from the heap.
 *
 * @param heap Pointer to the MaxHeap.
 * @return The Swap with the highest delta.
 */
bool extract_max(SwapHeap *heap, Swap *best);

/**
 * Invalidate Swaps in the heap if either male_node1 or male_node2 matches
 * the given node. Invalidated swaps are removed from the heap.
 *
 * @param heap Pointer to the MaxHeap.
 * @param node_to_invalidate The node to invalidate.
 */
void invalidate_swaps(SwapHeap *heap, int node_to_invalidate);

/**
 * Check if there are no positive delta values in the heap.
 *
 * @param heap Pointer to the MaxHeap.
 * @return True if all delta values are zero or negative, false otherwise.
 */
bool no_positive_deltas(SwapHeap *heap);

/**
 * Free the memory associated with the heap.
 *
 * @param heap Pointer to the MaxHeap.
 */
void destroy_swap_heap(SwapHeap *heap);