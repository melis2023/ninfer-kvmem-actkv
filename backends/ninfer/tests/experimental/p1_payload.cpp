// The existing independent byte-layout oracle also checks stale generations,
// fragmented physical mappings, transfer groups, and split-plane restoration.
#define main existing_paged_kv_test_main
#include "../test_kv_cache.cpp"
#undef main

int main() {
    try {
        int failures = existing_paged_kv_test_main();
        if (failures) { return failures; }
        ninfer::DeviceContext context(0);
        for (bool quantized : {false, true}) {
            ninfer::KVPageGeometry geometry;
            geometry.device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor;
            for (int layer = 0; layer < 3; ++layer) {
                geometry.planes.push_back({quantized ? ninfer::DType::I8 : ninfer::DType::BF16, 256, 4, 256});
                geometry.planes.push_back({quantized ? ninfer::DType::I8 : ninfer::DType::BF16, 256, 4, 256});
                if (quantized) {
                    geometry.planes.push_back({ninfer::DType::FP16, 4, 4, 256});
                    geometry.planes.push_back({ninfer::DType::FP16, 4, 4, 256});
                }
            }
            failures += exercise_layout_and_transfer(context, geometry,
                quantized ? "P1 3-layer INT8 K/V/scales" : "P1 3-layer BF16 K/V");
        }
        ninfer::KVPageGeometry rotated;
        rotated.device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor;
        for (int layer = 0; layer < 3; ++layer) {
            rotated.planes.push_back({ninfer::DType::I8, 256, 4, 256});
            rotated.planes.push_back({ninfer::DType::U8, 128, 4, 256});
            rotated.planes.push_back({ninfer::DType::FP16, 4, 4, 256});
            rotated.planes.push_back({ninfer::DType::FP16, 8, 4, 256});
        }
        failures += exercise_layout_and_transfer(context, rotated, "P10 3-layer RK8V4 K/V/scales");
        for (const auto order : {ninfer::PagedKVPlaneOrder::PageMajor, ninfer::PagedKVPlaneOrder::HeadMajor}) {
            ninfer::KVPageGeometry nvfp4;
            nvfp4.device_plane_order = order;
            // Independent stored schema: three Main layers followed by one MTP
            // layer, each with packed K/V and raw E4M3 group-16 scale bytes.
            for (int layer = 0; layer < 4; ++layer) {
                nvfp4.planes.push_back({ninfer::DType::U8, 128, 4, 256});
                nvfp4.planes.push_back({ninfer::DType::U8, 128, 4, 256});
                nvfp4.planes.push_back({ninfer::DType::U8, 16, 4, 256});
                nvfp4.planes.push_back({ninfer::DType::U8, 16, 4, 256});
            }
            failures += exercise_layout_and_transfer(context, nvfp4,
                order == ninfer::PagedKVPlaneOrder::PageMajor
                    ? "NVFP4 Main+MTP page-major codes/scales" : "NVFP4 Main+MTP head-major codes/scales");
        }
        for (const auto order : {ninfer::PagedKVPlaneOrder::PageMajor, ninfer::PagedKVPlaneOrder::HeadMajor}) {
            ninfer::KVPageGeometry k8v4;
            k8v4.device_plane_order = order;
            // Independent mixed schema: three Main layers and one MTP layer.
            // K has FP8 codes and one FP16 row scale; V has NVFP4 packed codes
            // and sixteen raw E4M3 group scales. Never assume equal scale planes.
            for (int layer = 0; layer < 4; ++layer) {
                k8v4.planes.push_back({ninfer::DType::FP8_E4M3FN, 256, 4, 256});
                k8v4.planes.push_back({ninfer::DType::U8, 128, 4, 256});
                k8v4.planes.push_back({ninfer::DType::FP16, 1, 4, 256});
                k8v4.planes.push_back({ninfer::DType::U8, 16, 4, 256});
            }
            failures += exercise_layout_and_transfer(context, k8v4,
                order == ninfer::PagedKVPlaneOrder::PageMajor
                    ? "K8V4 Main+MTP page-major codes/scales" : "K8V4 Main+MTP head-major codes/scales");
        }
        for (const auto order : {ninfer::PagedKVPlaneOrder::PageMajor, ninfer::PagedKVPlaneOrder::HeadMajor}) {
            for (const std::string format : {"fp8", "rk4v4", "rk4v4-e8", "rk2v4-e8"}) {
                ninfer::KVPageGeometry geometry;
                geometry.device_plane_order = order;
                const bool fp8 = format == "fp8";
                const int key_bytes = format == "rk2v4-e8" ? 64 : 128;
                // Independent stored schema, with three Main layers plus one MTP
                // layer. E8 root codes occupy two bytes per eight key dimensions.
                for (int layer = 0; layer < 4; ++layer) {
                    geometry.planes.push_back({fp8 ? ninfer::DType::FP8_E4M3FN : ninfer::DType::U8,
                        fp8 ? 256 : key_bytes, 4, 256});
                    geometry.planes.push_back({fp8 ? ninfer::DType::FP8_E4M3FN : ninfer::DType::U8,
                        fp8 ? 256 : 128, 4, 256});
                    geometry.planes.push_back({ninfer::DType::FP16, fp8 ? 1 : 4, 4, 256});
                    geometry.planes.push_back({ninfer::DType::FP16, fp8 ? 1 : 8, 4, 256});
                }
                const auto label = format + " Main+MTP " +
                    (order == ninfer::PagedKVPlaneOrder::PageMajor ? "page-major" : "head-major");
                failures += exercise_layout_and_transfer(context, geometry, label.c_str());
            }
        }
        std::cout << (failures ? "P1_PAYLOAD FAIL\n" : "P1_PAYLOAD PASS\n");
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "P1_PAYLOAD FAIL " << e.what() << '\n';
        return 1;
    }
}
