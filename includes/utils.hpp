#ifndef __SAMPLE_PARSER_HPP__
#define __SAMPLE_PARSER_HPP__

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string_view>
template <typename float_type> struct Args {
  const std::size_t n_samples;
  const float_type a;
  const float_type b;

  static Args parse(int argc, char **argv);
};

template <typename float_type>
Args<float_type> Args<float_type>::parse(int argc, char **argv) {
  std::size_t n_samples = 1000;
  float_type a = 0.0;
  float_type b = 1.0;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-n" && i + 1 < argc) {
      n_samples = static_cast<std::size_t>(std::stof(argv[++i]));
    } else if (std::string(argv[i]) == "-a" && i + 1 < argc) {
      a = std::stof(argv[++i]);
    } else if (std::string(argv[i]) == "-b" && i + 1 < argc) {
      b = std::stof(argv[++i]);
    } else {
      std::cerr << "Usage: " << argv[0] << " [-n n_samples] [-a a] [-b b]\n";
      exit(EXIT_FAILURE);
    }
  }

  return Args{n_samples, a, b};
}

template <typename view_type>
void save_csv(view_type samples, std::string_view name) {
  auto samples_host(Kokkos::create_mirror_view(Kokkos::HostSpace(), samples));
  Kokkos::deep_copy(samples_host, samples);

  const std::size_t n_samples = samples.extent(0);
  std::ofstream out(name.data());
  for (size_t i = 0; i < n_samples; ++i) {
    out << samples_host(i) << "\n";
  }
  out.close();
}

#endif
