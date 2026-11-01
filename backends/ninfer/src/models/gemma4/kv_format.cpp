#include "models/gemma4/kv_format.h"

#include <stdexcept>

namespace ninfer::models::gemma4 {

KvFormat kv_format(KvCacheStorage storage) {
    constexpr KvPlane f16{KvCoding::F16, false};
    const auto rot   = [](KvCoding c) { return KvPlane{c, true}; };
    const auto plain = [](KvCoding c) { return KvPlane{c, false}; };
    switch (storage) {
    case KvCacheStorage::BFloat16: return {storage, f16, f16, f16};
    case KvCacheStorage::Int8Group64:
        return {storage, rot(KvCoding::Int8G64), plain(KvCoding::Int8G64), plain(KvCoding::Int8G64)};
    case KvCacheStorage::Fp8E4M3Row256:
        return {storage, rot(KvCoding::Fp8Row), plain(KvCoding::Fp8Row), plain(KvCoding::Fp8Row)};
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return {storage, rot(KvCoding::Int8G64), plain(KvCoding::Int4G32), plain(KvCoding::Int8G64)};
    case KvCacheStorage::Nvfp4Group16:
        return {storage, rot(KvCoding::Nvfp4G16), rot(KvCoding::Nvfp4G16), plain(KvCoding::Nvfp4G16)};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {storage, rot(KvCoding::Fp8Row), rot(KvCoding::Nvfp4G16), plain(KvCoding::Fp8Row)};
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return {storage, rot(KvCoding::Lloyd4), plain(KvCoding::Int4G32), plain(KvCoding::Int4G32)};
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        return {storage, rot(KvCoding::Int4E8), plain(KvCoding::Int4G32), plain(KvCoding::Int4G32)};
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
        return {storage, rot(KvCoding::RootE8), plain(KvCoding::Int4G32), plain(KvCoding::Int4G32)};
    }
    throw std::invalid_argument("gemma4: unknown KV-cache storage");
}

} // namespace ninfer::models::gemma4
