// boost/functional/hash.hpp 兼容垫片（2026-10-27）
// lightsfm 的 vector2d.hpp 依赖 boost::hash_value/hash_combine（仅用于
// std::hash<Vector2d> 特化——本工程不使用 Vector2d 作为哈希 key）。
// 垫片按 boost 官方同款公式实现，lightsfm 源码零修改。
#ifndef BOOST_FUNCTIONAL_HASH_SHIM_20261027
#define BOOST_FUNCTIONAL_HASH_SHIM_20261027

#include <cstddef>
#include <functional>

inline void boost_hash_combine_impl(std::size_t& seed, std::size_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

namespace boost {

inline std::size_t hash_value(double v) {
  return std::hash<double>()(v);
}

inline std::size_t hash_value(float v) {
  return std::hash<float>()(v);
}

inline std::size_t hash_value(int v) {
  return std::hash<int>()(v);
}

inline std::size_t hash_value(std::size_t v) {
  return std::hash<std::size_t>()(v);
}

template <typename T>
inline void hash_combine(std::size_t& seed, T const& value) {
  boost_hash_combine_impl(seed, static_cast<std::size_t>(
                                    std::hash<T>()(value)));
}

}  // namespace boost

#endif
