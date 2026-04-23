# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/MoltenVK")
  file(MAKE_DIRECTORY "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/MoltenVK")
endif()
file(MAKE_DIRECTORY
  "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/moltenvk-prefix/src/moltenvk-build"
  "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/moltenvk-prefix"
  "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/moltenvk-prefix/tmp"
  "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/moltenvk-prefix/src/moltenvk-stamp"
  "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/moltenvk-prefix/src"
  "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/moltenvk-prefix/src/moltenvk-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/moltenvk-prefix/src/moltenvk-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/anthonygarcia-najera/Documents/GitHub/rpcs3/3rdparty/MoltenVK/moltenvk-prefix/src/moltenvk-stamp${cfgdir}") # cfgdir has leading slash
endif()
