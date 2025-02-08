#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "heap.h"

// Helper macros for parent and child indices
#define PARENT(i) ((i - 1) / 2)
#define LEFT(i)   (2 * i + 1)
#define RIGHT(i)  (2 * i + 2)

// Helper function to swap two Swap elements
static void swap_elements(Swap *a, Swap *b) {
    Swap temp = *a;
    *a = *b;
    *b = temp;
}

// Heapify up after insertion
static void heapify_up(SwapHeap *heap, int index) {
    while (index > 0) {
        int parent = PARENT(index);
        if (heap->data[parent].delta >= heap->data[index].delta) {
            break;
        }
        swap_elements(&heap->data[parent], &heap->data[index]);
        index = parent;
    }
}

// Heapify down after extraction or replacement
static void heapify_down(SwapHeap *heap, int index) {
    int size = heap->size;
    while (true) {
        int largest = index;
        int left = LEFT(index);
        int right = RIGHT(index);

        // Compare with left child
        if (left < size && heap->data[left].delta > heap->data[largest].delta) {
            largest = left;
        }

        // Compare with right child
        if (right < size && heap->data[right].delta > heap->data[largest].delta) {
            largest = right;
        }

        // If largest is not the current index, swap and continue
        if (largest != index) {
            swap_elements(&heap->data[index], &heap->data[largest]);
            index = largest;
        } else {
            break;
        }
    }
}

SwapHeap *create_swap_heap(int capacity) {
    if (capacity <= 0) {
        return NULL;
    }

    SwapHeap *heap = malloc(sizeof(SwapHeap));
    if (!heap) {
        return NULL;
    }

    heap->data = malloc(capacity * sizeof(Swap));
    if (!heap->data) {
        free(heap);
        return NULL;
    }

    heap->capacity = capacity;
    heap->size = 0;
    return heap;
}

void insert_swap(SwapHeap *heap, Swap swap) {
    if (!heap || heap->size >= heap->capacity) {
        return;
    }

    // If heap is full, check if new swap should replace smallest
    if (heap->size == heap->capacity) {
        // Find index of swap with lowest delta
        int min_index = 0;
        for (int i = 1; i < heap->size; i++) {
            if (heap->data[i].delta < heap->data[min_index].delta) {
                min_index = i;
            }
        }

        // If new swap has higher delta, replace smallest
        if (swap.delta > heap->data[min_index].delta) {
            heap->data[min_index] = swap;
            heapify_down(heap, min_index);
            heapify_up(heap, min_index);
        }
        return;
    }

    // Insert new swap at end and heapify up
    heap->data[heap->size] = swap;
    heapify_up(heap, heap->size);
    heap->size++;
}

bool extract_max(SwapHeap *heap, Swap *best) {
    if (!heap || heap->size == 0) {
        return false;
    }

    // Copy top element (max delta)
    *best = heap->data[0];

    // Move last element to top and reduce size
    heap->data[0] = heap->data[heap->size - 1];
    heap->size--;

    // Restore heap property
    if (heap->size > 0) {
        heapify_down(heap, 0);
    }

    return true;
}

void invalidate_swaps(SwapHeap *heap, int node_to_invalidate) {
    if (!heap) {
        return;
    }

    // Iterate through heap, mark and remove invalid swaps
    for (int i = 0; i < heap->size; ) {
        if (heap->data[i].nodeM1 == node_to_invalidate || 
            heap->data[i].nodeM2 == node_to_invalidate) {
            // Replace with last element
            heap->data[i] = heap->data[heap->size - 1];
            heap->size--;

            // Heapify to maintain heap property
            if (i < heap->size) {
                heapify_down(heap, i);
                heapify_up(heap, i);
            }
        } else {
            i++;
        }
    }
}

bool no_positive_deltas(SwapHeap *heap) {
    if (!heap) {
        return true;
    }

    for (int i = 0; i < heap->size; i++) {
        if (heap->data[i].delta > 0) {
            return false;
        }
    }
    return true;
}

void destroy_swap_heap(SwapHeap *heap) {
    if (!heap) {
        return;
    }

    free(heap->data);
    free(heap);
}