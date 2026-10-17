cmake_minimum_required(VERSION 3.25)
project(ea_het_rsvd_repro LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures, e.g. 80 for A100/A800 or 89 for RTX 4090D")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(EA_HET_CUDA_LIBS
    CUDA::cudart
    CUDA::cusparse
    CUDA::cusolver
    CUDA::cublas
    CUDA::curand)

function(add_ea_cuda_target target source)
  add_executable(${target} ${source})
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  target_link_libraries(${target} PRIVATE ${EA_HET_CUDA_LIBS} Threads::Threads)
endfunction()

add_ea_cuda_target(rsvd_mm_benchmark src/cuda/rsvd_mm_benchmark.cu)
add_ea_cuda_target(gpu_sparse_reference src/cuda/gpu_sparse_reference.cu)
add_ea_cuda_target(exact_noprobe src/cuda/rsvd_mm_exact_noprobe.cu)
add_ea_cuda_target(full_energy_probe src/cuda/rsvd_mm_full_energy_probe.cu)
add_ea_cuda_target(cached_reuse benchmarks/manuscript_completion/cached_reuse.cu)
add_ea_cuda_target(accumulation_stress src/cuda/accumulation_stress.cu)
add_ea_cuda_target(atq_device_tree_partitioned src/cuda/atq_device_tree_partitioned.cu)
add_ea_cuda_target(end_to_end_overlap src/cuda/end_to_end_overlap.cu)
add_ea_cuda_target(dual_lane_dynamic_overlap src/cuda/dual_lane_dynamic_overlap_residual.cu)

foreach(mode RANGE 0 2)
  add_ea_cuda_target(controller_precision_${mode} src/cuda/controller_precision_replay.cu)
  target_compile_definitions(controller_precision_${mode} PRIVATE PRECISION_MODE=${mode})
endforeach()
