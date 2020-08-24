/**
 * @file   cpu.cpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * Median GAR, CPU kernel implementation.
**/

#include <cmath>
#include <limits>

#include <common.hpp>
#include <array.hpp>
#include <threadpool.hpp>
#include <operations.hpp>
#include "decl.hpp"

// -------------------------------------------------------------------------- //
// Kernel implementation
namespace OP_NAME {

template<class T> class Kernel<CPUDevice, T>: public Static {
public:
    static void process(OpKernelContext& context [[gnu::unused]], size_t const n, size_t const d, Tensor const& input, Tensor& outpt) {
        // Median coordinate-by-coordinate
        Array<Array<T const>> grads{input.flat<T>().data(), {n, d}};
        T* output = outpt.flat<T>().data();
        parallel_for(pool, 0, d, [&](size_t start, size_t stop) {
            for (size_t x = start; x < stop; ++x) { // Coordinates to work on
                typename decltype(grads)::Iter axis, aend;
                ::std::tie(axis, aend) = grads.axis(0, x);
                auto length = aend - axis;
                auto copy = vlarray<T>(length); // Fine since not many threads (< 100)
                auto target = copy.get();
                for (auto source = axis; source < aend; ++source) { // Copy and filter out non-finite values
                    auto&& value = *source;
                    if (::std::isfinite(value)) {
                        *(target++) = value;
                    } else {
                        --length;
                    }
                }
                auto median = copy.get() + length / 2ul; // Moved 'median' down, so non-finite entries should rank highest
                ::std::nth_element(copy.get(), median, copy.get() + length, [&](T x, T y) {
                    if (::std::isfinite(x)) {
                        if (::std::isfinite(y))
                            return x < y;
                        return true;
                    } else {
                        return false;
                    }
                });
                output[x] = *median;
            }
        });
    }
};

// Explicit instantiations
template class Kernel<CPUDevice, float>;
template class Kernel<CPUDevice, double>;

}
