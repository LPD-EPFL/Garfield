/**
 * @file   string_view.hpp
 * @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2019 École Polytechnique Fédérale de Lausanne (EPFL).
 * See LICENSE file.
 *
 * @section DESCRIPTION
 *
 * C++ version-aware string_view include.
**/

#pragma once

// External headers
#if __cplusplus >= 201703L
    #include <string_view>
#else
    #include <experimental/string_view>
namespace std { using string_view = experimental::string_view; }
#endif
