#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/fixture.h"
#include "core/weight_view.h"

#include <algorithm>
#include <array>
#include <bit>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void file_set_and_bindings() {
    Fixture fixture;
    fixture.write(true);
    Reader reader(fixture.entry);
    const auto bytes = reader.read_object(reader.find("q5"));
    require(bytes.size() == 528 &&
                std::equal(bytes.begin(), bytes.end(), fixture.payload.begin() + 256),
            "cross-file parent bytes changed");
    require(reader.read_range(1344, 0).empty(), "empty terminal range failed");
    rejects([&] { (void)reader.read_range(1344, 1); }, "out-of-range read accepted");
    rejects([&] { (void)reader.geometry(reader.find("unused")); },
            "unknown selected encoding accepted");
    rejects([&] { (void)reader.read_range(1280, 64); }, "required missing part accepted");

    Binder binder(reader);
    const auto full = binder.parameter("matrix", {2, 130});
    const auto row  = binder.parameter("row", {1, 130});
    require(full.binding.parts[0].object == row.binding.parts[0].object, "shared parent lost");
    rejects([&] { (void)binder.parameter("matrix", {1, 260}); },
            "whole-object shape was silently reshaped");
    const auto reshaped = binder.parameter("row", {130});
    require(reshaped.binding.elements == 130, "Part reshape changed logical elements");
    const auto context =
        binder.values(binder.use("row", "context").auxiliaries.at("input_divisor"));
    const auto query = binder.values(binder.use("row", "query").auxiliaries.at("input_divisor"));
    require(std::bit_cast<std::uint32_t>(query.scalar_f32()) == 0x40000000 &&
                std::bit_cast<std::uint32_t>(context.scalar_f32()) == 0x40400000,
            "Use values were mixed");
    require(binder.use("row", "query").activation_policy == ActivationPolicy::AllowA4 &&
                binder.use("row", "context").activation_policy == ActivationPolicy::AllowA8,
            "Use policy changed");
    (void)binder.resource("text", "tokenizer.json");
    auto plan = std::move(binder).finish();
    require(plan.device_objects.size() == 1 && plan.device_capacity(0) == 528 &&
                plan.host_objects.size() == 1,
            "selected parent deduplication or residency failed");
}

void geometry_and_views() {
    const std::array<std::uint16_t, 12> direct_words{
        0, 0x8000, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110, 0x4120};
    const auto* direct_bytes = reinterpret_cast<const std::byte*>(direct_words.data());
    const WeightParent direct_parent{weight_geometry(QType::BF16, QuantLayout::Contiguous,
                                                     std::array<std::uint64_t, 3>{2, 2, 3}),
                                     direct_bytes};
    const WeightView direct_view{{2, 3}, {{&direct_parent, 3, 6}, {&direct_parent, 6, 9}}};
    const auto direct = native_weight(direct_view);
    require(direct.n == 2 && direct.k == 3 && direct.padded_shape[1] == 3 &&
                direct.qdata == direct_bytes + 6 && direct.payload == direct_bytes &&
                direct.payload_bytes == sizeof(direct_words),
            "Direct reshape lost its logical shape, element offset or owning parent");
    Fixture fixture;
    WeightParent parent{weight_geometry(QType::Q5_G64_FP16, QuantLayout::RowSplit,
                                        std::array<std::uint64_t, 2>{2, 130}),
                        fixture.payload.data() + 256};
    const WeightView row{{1, 130}, {{&parent, 130, 260}}};
    const auto planes = weight_row_planes(row.parts.front());
    require(parent.geometry.bytes == 528 && planes.codes == parent.data + 128 &&
                planes.high == parent.data + 288 && planes.scales == parent.data + 520,
            "Q5 view was interpreted as a standalone payload");
    const auto weight = native_weight(row);
    rejects<std::invalid_argument>([&] { (void)native_weight(WeightView{{2, 65}, row.parts}); },
                                   "quantized view changed K without preserving encoding groups");
    require(weight.qdata == planes.codes && weight.qhigh == planes.high &&
                weight.scales == planes.scales,
            "native RowSplit view lost its planes");
    require(std::to_integer<int>(planes.codes[0]) == 0x52 &&
                std::to_integer<int>(planes.scales[1]) == 0x40,
            "row view selected the wrong represented values");

    std::array<std::byte, 264> q2_bytes{};
    WeightParent q2_parent{weight_geometry(QType::Q2_G64_FP16, QuantLayout::RowSplit,
                                           std::array<std::uint64_t, 2>{2, 65}),
                           q2_bytes.data()};
    const WeightView q2_row{{1, 65}, {{&q2_parent, 65, 130}}};
    const auto q2 = native_weight(q2_row);
    require(q2_parent.geometry.bytes == q2_bytes.size() &&
                q2_parent.geometry.code_bytes == 64 && q2_parent.geometry.scale_offset == 256 &&
                q2.qdata == q2_bytes.data() + 32 && q2.scales == q2_bytes.data() + 260 &&
                q2.qhigh == nullptr && q2.padded_shape[1] == 128 && q2.group_size == 64,
            "Q2 row view lost its offset-code or scale plane");

    std::vector<std::byte> fp8(16896);
    WeightParent fp8_parent{weight_geometry(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale,
                                            std::array<std::uint64_t, 2>{256, 64}),
                            fp8.data()};
    WeightView fp8_rows{{128, 64}, {{&fp8_parent, 0, 128 * 64}}};
    require(weight_row_planes(fp8_rows.parts.front()).scales == fp8.data() + 16384,
            "FP8 view recomputed the parent scale base");
    rejects<std::invalid_argument>([&] { (void)native_weight(fp8_rows); },
                                   "complete-parent ABI accepted an FP8 submatrix");

    std::array<std::byte, 486> interleaved{};
    WeightParent row_parent{weight_geometry(QType::FP8_E4M3FN_ROW_FP16,
                                            QuantLayout::RowInterleaved,
                                            std::array<std::uint64_t, 2>{3, 160}),
                            interleaved.data()};
    const WeightView row_view{{2, 160}, {{&row_parent, 160, 480}}};
    const auto row_fp8 = native_weight(row_view);
    require(row_parent.geometry.bytes == interleaved.size() &&
                row_parent.geometry.code_bytes == 480 && row_parent.geometry.scale_bytes == 6 &&
                weight_scale_offset(row_parent.geometry, 2, 0) == 484 &&
                row_fp8.qdata == interleaved.data() + 162 &&
                row_fp8.scales == interleaved.data() + 322 &&
                row_fp8.scale_dtype == DType::FP16 && row_fp8.scale_nb[0] == 162,
            "interleaved FP8 row view lost its row stride or FP16 scale");
    rejects<std::invalid_argument>([&] {
        (void)weight_geometry(QType::FP8_E4M3FN_ROW_FP16, QuantLayout::RowScale,
                              std::array<std::uint64_t, 2>{3, 160});
    }, "FP16 row scales admitted in the BF16 plane layout");

    std::vector<std::byte> nvfp4(4612);
    WeightParent nv_parent{weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                           std::array<std::uint64_t, 2>{128, 64}),
                           nvfp4.data(), 2.0F};
    require(weight_scale_offset(nv_parent.geometry, 31, 0) == 4096 + 496 &&
                weight_scale_offset(nv_parent.geometry, 32, 0) == 4096 + 4,
            "NVFP4 parent swizzle origin was lost");
    const auto nv_rows = weight_row_planes({&nv_parent, 31 * 64, 33 * 64});
    require(nv_rows.swizzled_scales && nv_rows.row_begin == 31 &&
                nv_rows.scales == nvfp4.data() + 4096,
            "NVFP4 submatrix lost its parent coordinates");
    const WeightView complete{{128, 64}, {{&nv_parent, 0, 128 * 64}}};
    const auto query   = native_weight(complete, 3.0F);
    const auto context = native_weight(complete, 4.0F);
    require(query.weight_scale_divisor == 2 && query.input_scale_divisor == 3 &&
                context.input_scale_divisor == 4 && nv_parent.weight_scale_divisor == 2,
            "per-use native parameters changed the parent");
}

void invalid_directories() {
    Fixture fixture;
    const auto bad = [&](auto mutate) {
        Json root = fixture.root;
        mutate(root);
        rejects([&] { (void)parse_directory(root, "model.ninfer"); }, "invalid directory accepted");
    };
    bad([](Json& root) { root["bindings"]["row"]["parts"][0]["range"] = {129, 261}; });
    bad([](Json& root) { root["uses"].push_back(root["uses"][0]); });
    bad([](Json& root) { root["objects"][1]["offset"] = 4; });
    bad([](Json& root) { root["objects"][2]["id"] = "q5"; });
    bad([](Json& root) { root["files"][1]["path"] = "../other"; });
    bad([](Json& root) { root["files"][1]["payload_bytes"] = 1.5; });
    bad([](Json& root) { root["components"]["text"]["resources"]["tokenizer.json"] = "q5"; });
    bad([](Json& root) { root["components"] = Json::object(); });
    rejects([] { (void)parse_json("{\"a\":{\"x\":1,\"x\":2}}", "duplicate"); },
            "duplicate JSON key accepted");
    rejects([] { (void)parse_json("{\"x\":1,\"x\":2}", "duplicate"); },
            "duplicate root key accepted");
    rejects([] { (void)parse_json("[{\"x\":1,\"\\u0078\":2}]", "duplicate"); },
            "escaped duplicate in an array accepted");
    rejects([] { (void)parse_json("{\"x\":1} {\"y\":2}", "trailing"); },
            "trailing JSON value accepted");
    const auto siblings = parse_json("{\"a\":[{\"x\":1},{\"x\":2}],\"x\":3}", "siblings");
    require(siblings["a"][1]["x"] == 2 && siblings["x"] == 3,
            "keys in separate objects must remain independent");
    Json wide = Json::object();
    for (int i = 0; i < 32000; ++i) { wide["expert-" + std::to_string(i)] = {{"object", i}}; }
    require(parse_json(wide.dump(), "wide") == wide, "wide binding map changed in parsing");
    fixture.write();
    {
        std::fstream file(fixture.entry, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(7);
        file.put(2);
    }
    rejects([&] { Reader reader(fixture.entry); }, "v2 magic accepted");
    fixture.write();
    {
        std::fstream file(fixture.directory / "weights-second.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(16);
        file.put(0);
    }
    // A mapped file cannot be truncated or resized while its section is open (Windows returns
    // ERROR_USER_MAPPED_FILE), so every rewrite happens with no Reader alive.
    {
        Reader wrong_part(fixture.entry);
        rejects([&] { (void)wrong_part.read_object(wrong_part.find("q5")); },
                "foreign continuation accepted");
    }
    fixture.write();
#ifdef _WIN32
    std::filesystem::resize_file(fixture.entry, std::filesystem::file_size(fixture.entry) - 1);
    rejects([&] {
        Reader shortened(fixture.entry);
        (void)shortened.read_range(499, 1);
    }, "truncated entry was accepted");
#else
    // POSIX permits resizing under an open mapping, so the read-time EOF check runs there.
    Reader shortened(fixture.entry);
    std::filesystem::resize_file(fixture.entry, std::filesystem::file_size(fixture.entry) - 1);
    rejects([&] { (void)shortened.read_range(499, 1); }, "premature EOF accepted");
#endif
}

} // namespace

int main(int argc, char** argv) {
    try {
        file_set_and_bindings();
        geometry_and_views();
        invalid_directories();
        // Optional production-writer fixture or explicitly selected real artifact.
        if (argc == 2) {
            Reader reader(argv[1]);
            std::uint64_t bytes = 0;
            for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
                reader.validate_object({i});
                bytes =
                    checked_add(bytes, object_bytes(reader.directory().objects[i]), "object bytes");
            }
            std::cout << "v3 objects=" << reader.directory().objects.size() << " bytes=" << bytes
                      << '\n';
        }
        std::cout << "artifact reader and binding checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
