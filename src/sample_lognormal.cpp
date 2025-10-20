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

    Ftype mu = 0.0;    // Mean of the log distribution
    Ftype sigma = 1.0; // Standard deviation of the log distribu
    auto target = KOKKOS_LAMBDA(const Ftype x) {

      if (x <= 0.0) {
        return Ftype(0.0);
      }
      Ftype coefficient = 1.0 / (x * sigma * std::sqrt(2.0 * M_PI));
      Ftype exponent = -std::pow(std::log(x) - mu, 2) / (2.0 * sigma * sigma);

      return coefficient * std::exp(exponent);
    };
    Kokkos::View<Ftype *, Kokkos::DefaultExecutionSpace> samples(
        "samples", args.n_samples);

    rc = Sampling::metropolis(target, samples, args.a, args.b);
    if (rc == 0) {
      save_csv(samples, "samples_lognormal.csv");
    }
  }
  Kokkos::finalize();
  return rc;
}
