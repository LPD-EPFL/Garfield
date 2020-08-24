/**
 * @file   constexpr.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Constexpr helpers.
**/

#pragma once

// External headers
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ConstExpr {
// -------------------------------------------------------------------------- //
// Min/max helpers

/** Return the minimum of the given parameters
 * @param ... Parameters
 * @return Minimum of these parameters
**/
template<class A> constexpr inline auto min(A&& a) {
    return a;
}
template<class A, class B> constexpr inline auto min(A&& a, B&& b) {
    return a < b ? a : b;
}
template<class A, class B, class... C> constexpr inline auto min(A&& a, B&& b, C&&... c) {
    return a < b ? min(a, ::std::forward<C>(c)...) : min(b, ::std::forward<C>(c)...);
}

/** Return the maximum of the given parameters
 * @param ... Parameters
 * @return Maximum of these parameters
**/
template<class A> constexpr inline auto max(A&& a) {
    return a;
}
template<class A, class B> constexpr inline auto max(A&& a, B&& b) {
    return a < b ? b : a;
}
template<class A, class B, class... C> constexpr inline auto max(A&& a, B&& b, C&&... c) {
    return a < b ? max(b, ::std::forward<C>(c)...) : max(a, ::std::forward<C>(c)...);
}

// -------------------------------------------------------------------------- //
// Alignment helpers

/** Align the given size/address/pointer.
 * @param a Given size/address/pointer
 * @param b Alignment requirement (optional if pointer given)
 * @param o Additional offset (optional)
 * @return Aligned size/address/pointer
**/
template<class I> constexpr inline I align(I a, size_t b, I o = 0) {
    return (a + o + b - 1) / b * b;
}
template<class P> constexpr inline P* align(P* a, size_t b = alignof(P), uintptr_t o = 0) {
    return reinterpret_cast<P*>((reinterpret_cast<uintptr_t>(a) + o + b - 1) / b * b);
}

// -------------------------------------------------------------------------- //
// Iterable helpers

/** Naively checks whether a given element belongs to some iterable.
 * @param x Element to check
 * @param s Given iterable
 * @return Whether x == <some element in s>
**/
template<class X, class S> constexpr inline bool is_in(X&& x, S&& s) {
    for (auto&& i: s) {
        if (x == i)
            return true;
    }
    return false;
}

// -------------------------------------------------------------------------- //
}
