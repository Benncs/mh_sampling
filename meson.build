project(
  'kokkos-sampling',
  'cpp',
  version: '0.1.0',
  meson_version: '>= 1.3.0',
  default_options: ['warning_level=3', 'b_colorout=always', 'cpp_std=c++20'],
)

build_executable = get_option('build_exe')
public_includes = include_directories(f'public/')
kokkos_sampling = declare_dependency(
  include_directories: public_includes,
)

if build_executable
  subdir('src')
endif