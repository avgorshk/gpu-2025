#include <random>

// Xoshiro generator
struct FastRNG {
  uint64_t s[4];

  FastRNG(uint64_t seed = 0) {
    std::mt19937_64 seeder(seed);
    s[0] = seeder();
    s[1] = seeder();
    s[2] = seeder();
    s[3] = seeder();
  }

  uint64_t operator()() {
    const uint64_t result = std::rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = std::rotl(s[3], 45);
    return result;
  }
};

static FastRNG generator(std::random_device{}());

void random_fill(std::vector<float>& vec, float min_val, float max_val) {
  float range = max_val - min_val;
  for (size_t i = 0; i < vec.size(); i++) {
    vec[i] = (generator() / (float)UINT64_MAX) * range + min_val;
  }
}
