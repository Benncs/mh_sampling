#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_sampling/metropolis.hpp>
#include <utils.hpp>

using Ftype = float;
int main(int argc, char **argv) {

  Kokkos::initialize(argc, argv);

  int rc = 0;
  {
    auto args = Args<Ftype>::parse(argc, argv);
    Ftype lambda = Kokkos::log(2) / (2);
    auto target = KOKKOS_LAMBDA(const Ftype x) {
      return lambda * Kokkos::exp(-lambda * x);
    };
    Kokkos::View<Ftype *, Kokkos::DefaultExecutionSpace> samples(
        "samples", args.n_samples);

    rc = Sampling::metropolis(target, samples, args.a, args.b);
    if (rc == 0) {
      //      save_csv(samples, "samples_exp.csv");

      Ftype mean = 0;
      Kokkos::parallel_reduce(
          "mean_exp", Kokkos::RangePolicy<>(0, args.n_samples),
          KOKKOS_LAMBDA(const int i, Ftype &lmean) { lmean += samples(i); },
          mean);

      mean = mean / Ftype(args.n_samples);

      std::cout << "Mean expected: " << 1 / lambda << std::endl;
      std::cout << "Mean exp: " << mean << std::endl;
    }
  }
  Kokkos::finalize();
  return rc;
}
