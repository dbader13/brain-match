#pragma once
#include <stdint.h>

typedef struct xoshiro256ss_state{
    uint64_t s[4];
} xoshiro256ss_state;

/**
 * Initializes the xoshiro256** state with a given seed.
 * 
 * @param state A pointer to the xoshiro256ss_state struct to initialize.
 * @param seed The seed value to initialize the state with.
*/
void xoshiro256ss_init(struct xoshiro256ss_state *state, uint64_t seed);

/**
 * Generates a random 64-bit number using the xoshiro256** algorithm.
 * 
 * @param state A pointer to the xoshiro256ss_state struct.
 * @return A random 64-bit unsigned integer.
*/
uint64_t xoshiro256ss(struct xoshiro256ss_state *state);
