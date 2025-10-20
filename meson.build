project(
  'kokkos-sampling',
  'cpp',
  version: '0.1.0',
  meson_version: '>= 1.3.0',
  default_options: ['warning_level=3', 'b_colorout=always', 'cpp_std=c++20'],
)

use_cuda = get_option('use_cuda')
build_executable = get_option('build_exe')
outdir = meson.current_build_dir()
root = meson.project_source_root()
project_name = meson.project_name()
build_type = get_option('buildtype')
cpp = meson.get_compiler('cpp')
global_install_dir = get_option('prefix') + '/' + get_option('libdir')

name = f'@project_name@_app'
includes = include_directories(f'public/')
kokkos_sampling = declare_dependency(
  include_directories: includes,
)

if build_executable

  arch_specific_args = []
  compiler = meson.get_compiler('cpp')
  if compiler.has_argument('-fno-math-errno')
    arch_specific_args = [arch_specific_args, '-fno-math-errno']
  endif
  flto_flag = []
  if compiler.has_argument('-flto')
    flto_flag = [flto_flag, '-flto']
  endif
  if compiler.has_argument('-dlto')
    flto_flag = [flto_flag, '-dlto']
  endif
  if compiler.has_argument('-march=native')
    arch_specific_args = [arch_specific_args, '-march=native']
  endif
  if compiler.has_argument('-mtune=native')
    arch_specific_args = [arch_specific_args, '-mtune=native']
  endif
  if compiler.has_argument('-Wno-interference-size')
    # Disable cache line detection size
    arch_specific_args = [arch_specific_args, '-Wno-interference-size']
  endif
  compiler.has_argument('-fopenmp')

  ## Define main shared variables
  use_kokkos_tools = true

  cpp_args = [arch_specific_args]
  if (use_kokkos_tools)
    cpp_args = [cpp_args, '-DENABLE_KOKKOS_PROFILING']
  endif

  ## Get buildtype specific options

  CMAKE_TYPE = 'Release'
  if build_type == 'debug' or build_type == 'debugoptimized'
    # NOTHING
    CMAKE_TYPE = 'Debug'
    add_global_arguments('-DDEBUG', language: 'cpp')
  else
    add_global_arguments('-DNDEBUG', language: 'cpp')
    if compiler.has_argument('-ffast-math')
      arch_specific_args = [arch_specific_args, '-ffast-math']
    endif
  endif

  thread_dep = dependency('threads', required: true)
  omp = dependency('openmp')
  hwloc = dependency('hwloc')
  dl_dep = compiler.find_library('dl', required: true)
  use_system_kokkos = get_option('use_system_kokkos')
  cmake = import('cmake', required: true)

  if use_system_kokkos

    __cmake_kokkos_dep__ = dependency(
      'Kokkos',
      method: 'cmake',
      required: true,
      components: ['kokkos'],
      modules: ['Kokkos::kokkos'],
    )

    kokkos_compile_arg = ['-DKOKKOS_DEPENDENCE', '-Wno-unknown-pragmas']

    if use_cuda
      kokkos_compile_arg = [
        kokkos_compile_arg,
        '-extended-lambda',
        '-Wext-lambda-captures-this',
      ]

    endif

    kokkos = declare_dependency(
      dependencies: [__cmake_kokkos_dep__, omp, hwloc, thread_dep, dl_dep],
      compile_args: kokkos_compile_arg,
    )
  else
    kokkos_option = cmake.subproject_options()
    kokkos_option.add_cmake_defines({'CMAKE_CXX_STANDARD': '20'})
    kokkos_option.add_cmake_defines({'CMAKE_BUILD_TYPE': CMAKE_TYPE})
    kokkos_option.add_cmake_defines({'Kokkos_ENABLE_OPENMP': 'ON'})
    kokkos_option.add_cmake_defines({'Kokkos_ENABLE_HWLOC': 'ON'})
    kokkos_option.add_cmake_defines({'Kokkos_ARCH_NATIVE': 'ON'})
    kokkos_option.add_cmake_defines({'CMAKE_POSITION_INDEPENDENT_CODE': 'ON'})
    kokkos_option.set_install(false)
    sub_kokkos = cmake.subproject('kokkos', required: true, options: kokkos_option)

    kokkos = declare_dependency(
      dependencies: [
        sub_kokkos.dependency('kokkos'),
        sub_kokkos.dependency('kokkoscore'),
        sub_kokkos.dependency('kokkossimd'),
        sub_kokkos.dependency('kokkoscontainers'),
        sub_kokkos.dependency('kokkosalgorithms'),
        omp,
        hwloc,
        thread_dep,
        dl_dep,
      ],
    )
  endif

  add_project_arguments(cpp_args, language: 'cpp')
  privateincludes = include_directories('./includes')

  lib_app = library(
    f'@project_name@',
    'src/utils.cpp',
    include_directories: [includes,privateincludes],
    dependencies: [kokkos_sampling, kokkos],
    cpp_args: [],
  )

  dep_app = declare_dependency(
    include_directories: [includes,privateincludes],
    link_with: lib_app,
    dependencies: [kokkos_sampling, kokkos],
    compile_args: [],
  )

  executable(
    f'@project_name@_exp',
    f'src/sample_exp.cpp',
    dependencies: [dep_app],
    link_args: [flto_flag],
  )

  executable(
    f'@project_name@_lognormal',
    f'src/sample_lognormal.cpp',
    dependencies: [dep_app],
    link_args: [flto_flag],
  )
  run_command('sh', '-c', f'ln -s @outdir@/compile_commands.json .', check: false)

endif
