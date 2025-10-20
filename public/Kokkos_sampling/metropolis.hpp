#ifndef __SAMPLE_METROPOLIS_HPP__
#define __SAMPLE_METROPOLIS_HPP__

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#ifdef NDEBUG
#include <random>
#endif

namespace Sampling {

namespace {
template <typename float_type, typename generator_type>
  requires(std::same_as<float_type, float>)
KOKKOS_INLINE_FUNCTION float_type gen_samples(float_type a, float_type b,
                                              generator_type &gen) {
  return gen.frand(a, b);
}

template <typename float_type, typename generator_type>
  requires(std::same_as<float_type, double>)
KOKKOS_INLINE_FUNCTION float_type gen_samples(float_type a, float_type b,
                                              generator_type &gen) {
  return gen.drand(a, b);
}
} // namespace

/**
 * @brief Performs the Metropolis-Hastings algorithm to sample from a target
 * distribution.
 *
 * This function generates samples from a given target distribution using
 * the Metropolis-Hastings algorithm. It requires the target distribution
 * to be provided as a Kokkos callable (e.g., KOKKOS_LAMBDA or KOKKOS_FUNCTION),
 * and fills the provided samples view with generated values.
 *
 * @tparam lamda_type   Type of the callable representing the target
 * distribution. Must be callable with a value of type `float_type` and return a
 * probability or density.
 * @tparam view_type    Type of the container or view that holds the samples.
 *                      Its `value_type` must be `float_type`.
 * @tparam float_type   Floating point type used for computations and samples
 * (e.g., float, double).
 *
 * @requires std::same_as<typename view_type::value_type, float_type>
 *
 * @param target_distribution  Callable representing the target distribution.
 *                             Should be able to evaluate the distribution at
 * given points.
 * @param samples              View or container to hold the generated samples.
 *                             Must be mutable and have size preallocated.
 * @param a                    Lower bound of the sampling interval.
 * @param b                    Upper bound of the sampling interval.
 * @param seed                 Optional seed for the random number generator
 * (default is 0).
 *
 * @return int                 Status code indicating success or failure.
 *                             (Specify your convention here, e.g., 0 for
 * success)
 */
template <typename lamda_type, typename view_type, typename float_type>
  requires(std::same_as<typename view_type::value_type, float_type>)
int metropolis(const lamda_type &target_distribution, view_type samples,
               float_type a, float_type b, uint64_t seed = 0) {

  using device_type = view_type::device_type;
  if (seed == 0) {
#ifndef NDEBUG
    seed = 2025;
#else
    seed = std::random_device{}();
#endif
  }

  if (b <= a) {
    std::cerr << "Sample bounds must be in [a,b] with a<b" << std::endl;
    return 1;
  }

  const std::size_t n_samples = samples.extent(0);
  auto random_pool = Kokkos::Random_XorShift1024_Pool<>(seed);

  Kokkos::View<float_type, device_type> x_t("x_t");
  Kokkos::deep_copy(x_t, (a + b) / 2);

  Kokkos::parallel_for(
      "sampling", Kokkos::RangePolicy<>(0, n_samples),
      KOKKOS_LAMBDA(const int i) {
        auto gen = random_pool.get_state();

        const float_type u = gen_samples<float_type>(0, 1, gen);
        const float_type x_prime = gen_samples(a, b, gen);
        const float_type alpha =
            target_distribution(x_prime) / target_distribution(x_t());
        const float_type x_new = x_prime * (u <= alpha) + x_t() * (u > alpha);
        Kokkos::atomic_exchange(&x_t(), x_new);
        samples(i) = x_t();
        random_pool.free_state(gen);
      });
  return 0;
}

template <typename lamda_type, typename float_type, std::size_t N>
Kokkos::View<float_type[N]>
metropolis_fixed_size(const lamda_type &target_distribution, float_type a,
                      float_type b, uint64_t seed = 0) {
  Kokkos::View<float_type[N]> samples("samples");
  auto _ = metropolis(target_distribution, samples, a, b, seed);
  return samples;
}

template <typename lamda_type, typename float_type>
Kokkos::View<float_type *>
metropolis_dyn_size(const lamda_type &target_distribution, float_type a,
                    float_type b, std::size_t n_samples, uint64_t seed = 0) {
  Kokkos::View<float_type *> samples("samples", n_samples);
  auto _ = metropolis(target_distribution, samples, a, b, seed);
  return samples;
}

} // namespace Sampling
#endif
