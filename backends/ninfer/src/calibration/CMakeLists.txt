# Device calibration: times the route families a device profile can steer on the current GPU.
# route_catalog.cpp (host) lists the keys and candidate schedules, and their digest.
add_library(ninfer_calibration STATIC device_calibration.cu route_catalog.cpp)
ninfer_internal_includes(ninfer_calibration)
ninfer_cuda_archive(ninfer_calibration)
target_link_libraries(ninfer_calibration PUBLIC ninfer_ops ninfer_core)
