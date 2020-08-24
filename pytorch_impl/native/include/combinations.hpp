/**
 * @file   combination.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Iterator of all the subsets of size k among n.
**/

#pragma once

// Compiler version check
#if __cplusplus < 201304L
    #error This translation unit requires at least a C++14 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// External header
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

// Internal header
#include "common.hpp"

// -------------------------------------------------------------------------- //
// Combination iterator over the first 'n' natural integers

/** Combination "iterator" over the first 'n' natural integers class.
**/
class Combinations {
public:
    /** Natural number class alias.
    **/
    using Number = uint_least64_t;
    /** Index class alias.
    **/
    using Index = size_t;
private:
    /** Array of indexes class alias.
    **/
    using Combination = ::std::vector<Index>;
public:
    /** Count the total number of combinations.
     * @param n Total number of elements to choose from
     * @param k Number of elements to pick
     * @return n choose k, 0 in case of overflow/invalid input (only if debug enabled)
    **/
    static Number count(Number n, Number k) noexcept {
#ifndef NDEBUG
        // Assert parameter validity
        if (unlikely(n < k || k == 0))
            return 0;
#endif
        // Prepare counting
        auto r = Number{1};
        auto l = n - k;
        if (l < k)
            ::std::swap(l, k);
        auto d = Number{2};
        // Start counting
        for (auto i = l + 1; i <= n; ++i) {
            auto nr = r * i;
#ifndef NDEBUG
            // Check and abort if overflow
            if (unlikely(nr < r))
                return 0;
#endif
            r = nr;
            for (; d <= k && r % d == 0; ++d) // In-loop to "delay" overflow
                r /= d;
        }
        // Return count
        return r;
    }
private:
    Number       nbelements; // Number of elements to choose 'k' from
    Number   nbcombinations; // Total number of different combinations
    Combination combination; // Current combination
public:
    /** Default copy constructor/assignment.
    **/
    Combinations(Combinations const&) = default;
    Combinations& operator=(Combinations const&) = default;
    /** Default move constructor/assignment.
    **/
    Combinations(Combinations&&) = default;
    Combinations& operator=(Combinations&&) = default;
    /** Dimension constructor.
     * @param n Number of elements to select from
     * @param k Number of selected elements (must satisfies 0 < k ≤ n)
     * @throw 'overflow_error' if parameters are invalid/too large
    **/
    Combinations(Number const& n, Number const& k): nbelements{n}, nbcombinations{count(n, k)}, combination(k) {
#ifndef NDEBUG
        if (unlikely(nbcombinations == 0))
            throw ::std::overflow_error{"Combination parameters are too large or invalid"};
#endif
    }
public:
    /** Get the iterator parameters.
     * @return Iterator parameters
    **/
    auto get_parameters() const noexcept {
        return ::std::make_tuple(static_cast<Number>(nbelements), static_cast<Number>(combination.size()));
    }
    /** Get the total number of different combinations.
     * @return Total number of different combinations
    **/
    auto const& get_count() const noexcept {
        return nbcombinations;
    }
    /** Get the current (and changing) combination, state is undefined before the first call to 'seek'.
     * @return Current (and changing) combination, always a strictly increasing array of values
    **/
    auto const& get_current() const noexcept {
        return combination;
    }
public:
    /** Quickly seek a fixed position.
     * @param pos Position to seek, must be < total number of combinations
     * @throw 'overflow_error' (if debug enabled)/undefined behavior (if debug disabled) when seeking past the last position
    **/
    void seek(Number pos) {
#ifndef NDEBUG
        // Assert position validity
        if (unlikely(pos >= nbcombinations))
            throw ::std::overflow_error{"Seeking combination past the last position"};
#endif
        // Prepare seek
        auto const n = nbelements;
        auto const k = static_cast<Number>(combination.size());
        // Perform seek
        Index  v = 0;
        Number i = 0;
        for (; i < k - 1; ++i, ++v) { // For every position (except last one)
            // Find associated index
            while (true) {
                // Check current index
                auto a = count(n - v - 1, k - i - 1);
                if (pos < a)
                    break;
                // Else go to next one
                pos -= a;
                ++v;
            }
            // Set associated index
            combination[i] = v;
        }
        // Set associated index (last position)
        combination[i] = v + pos;
    }
    /** Quickly move to the next position, should be (on average) faster than calling 'seek' on the next position, undefined behavior if 'seek' has never been called.
     * @throw 'overflow_error' if called from the last position
    **/
    void next() {
        auto const k = combination.size();
        // Go through every position, last to first...
        Number i = k;
        while (i --> 0) {
            // ...until an increase-able position is found
            if (combination[i] < nbelements + i - k) {
                auto v = ++combination[i];
                for (auto j = i + 1; j < k; ++j)
                    combination[j] = v + j - i;
                return;
            }
        }
        throw ::std::overflow_error{"Next combination from the last position"};
    }
};
