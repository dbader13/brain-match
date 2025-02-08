#include <stdint.h>

// RNG code, taken from xoshiro256** implementation on Wikipedia 
struct xoshiro256ss_state {
    uint64_t s[4];
};

struct splitmix64_state {
	uint64_t s;
};

inline uint64_t splitmix64(struct splitmix64_state *state) {
	uint64_t result = (state->s += 0x9E3779B97f4A7C15);
	result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
	result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
	return result ^ (result >> 31);
}

void xoshiro256ss_init(struct xoshiro256ss_state *state, uint64_t seed) {
	struct splitmix64_state smstate = {seed};
	for(int i = 0; i < 4; i++){
		state->s[i] = splitmix64(&smstate);
	}
}

inline uint64_t rol64(uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro256ss(struct xoshiro256ss_state *state) {
    uint64_t *s = state->s;
	uint64_t const result = rol64(s[1] * 5, 7) * 9;
	uint64_t const t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;
	s[3] = rol64(s[3], 45);

	return result;
}