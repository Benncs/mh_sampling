#ifndef __SAMPLE_METROPOLIS_HPP__
#define __SAMPLE_METROPOLIS_HPP__

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#ifdef NDEBUG
#include <random>
#endif

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

template <typename lamda_type, typename view_type, typename float_type>
  requires(std::same_as<typename view_type::value_type, float_type>)
int metropolis(const lamda_type &target_distribution, view_type samples,
               float_type a, float_type b) {
#ifndef NDEBUG
  const uint64_t seed = 2025;
#else
  const uint64_t seed = std::random_device{}();
#endif

  if (b <= a) {
    std::cerr << "Sample bounds must be in [a,b] with a<b" << std::endl;
    return 1;
  }

  const std::size_t n_samples = samples.extent(0);
  auto random_pool = Kokkos::Random_XorShift1024_Pool<>(seed);

  Kokkos::View<float_type> x_t("x_t");
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

#endif
