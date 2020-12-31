/**
 * @file   optional.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * All rights reserved.
 *
 * @section DESCRIPTION
 *
 * C++ version-aware optional<T> include.
**/

#pragma once

// External headers
#if __cplusplus >= 201703L
    #include <optional>
#else
    #include <experimental/optional>
namespace std {
    template<class T> using optional = experimental::optional<T>;
    using experimental::nullopt;
}
#endif
