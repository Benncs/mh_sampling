#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_sampling/metropolis.hpp>
#include <utils.hpp>

using Ftype = float;

template <typename view_type> void test(view_type samples) {
  auto n_samples = samples.extent(0);
  Ftype mean = 0;
  Kokkos::parallel_reduce(
      "mean_normal", Kokkos::RangePolicy<>(0, n_samples),
      KOKKOS_LAMBDA(const int i, Ftype &lmean) { lmean += samples(i); }, mean);

  mean = mean / Ftype(n_samples);
  auto rel = (mean - 0) / 0 * 100;
  std::cout << "Normal with " << n_samples << " samples\r\n";
  std::cout << "Mean expected: " << 0 << std::endl;
  std::cout << "Mean exp: " << mean << std::endl;
  std::cout << "Rel error: " << rel << "%" << std::endl;
}

int main(int argc, char **argv) {

  Kokkos::initialize(argc, argv);

  int rc = 0;
  {
    auto args = Args<Ftype>::parse(argc, argv);
    const float mu = 0;
    const float sigma = 1.;
    auto target = KOKKOS_LAMBDA(const Ftype x) {
      return 1. / Kokkos::sqrt(2. * 3.14 * (sigma) * (sigma)) *
             Kokkos::exp((-(x - mu) * (x - mu) / (2. * sigma * sigma)));
    };

    auto samples =
        Sampling::metropolis_dyn_size(target, -5., 5., args.n_samples);

    test(samples);
    save_csv(samples, "samples_normal.csv");

    auto samples_fixed =
        Sampling::metropolis_fixed_size<decltype(target), float, 10000>(
            target, args.a, args.b);

    test(samples_fixed);
    //      save_csv(samples, "samples_exp.csv");
  }
  Kokkos::finalize();
  return rc;
}
