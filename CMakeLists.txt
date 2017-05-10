#
# Copyright (c) 2015-2016 CNRS
# Copyright (c) 2015 Wandercraft, 86 rue de Paris 91400 Orsay, France.
#
# This file is part of Pinocchio
# pinocchio is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# pinocchio is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Lesser Public License for more details. You should have
# received a copy of the GNU Lesser General Public License along with
# pinocchio If not, see
# <http://www.gnu.org/licenses/>.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/eigen.cmake)
INCLUDE(cmake/python.cmake)

SET(PROJECT_NAME pycado)
SET(PROJECT_DESCRIPTION "Python bindings for dedicated ACADO problems")
SET(PROJECT_URL "???")
OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" ON)

SET(DOXYGEN_USE_MATHJAX YES)

IF(APPLE)
  SET(CMAKE_MACOSX_RPATH TRUE)
  SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
     set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif("${isSystemDir}" STREQUAL "-1")
ENDIF(APPLE)

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

SETUP_PROJECT()

IF(WIN32)
  SET(LINK copy_if_different)
ELSE(WIN32)
  SET(LINK create_symlink)
ENDIF(WIN32)

# --- OPTIONS ----------------------------------------
OPTION (INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
OPTION (BUILD_PYTHON_INTERFACE "Build the python binding" ON)

IF (INITIALIZE_WITH_NAN)
  MESSAGE (STATUS "Initialize with NaN all the Eigen entries.")
  ADD_DEFINITIONS(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
ENDIF (INITIALIZE_WITH_NAN)

# ----------------------------------------------------
# --- DEPENDANCIES -----------------------------------
# ----------------------------------------------------
ADD_REQUIRED_DEPENDENCY("eigen3 >= 3.0.5")

# Spetcial care of Eigen3 greater than 3.2.10
IF(EIGEN3_FOUND)
  IF(${EIGEN3_VERSION} VERSION_GREATER "3.2.10")
    ADD_DEFINITIONS(-DEIGEN3_FUTURE)
    SET(EIGEN3_FUTURE TRUE)
    PKG_CONFIG_APPEND_CFLAGS("-DEIGEN3_FUTURE")
  ELSE(${EIGEN3_VERSION} VERSION_GREATER "3.2.10")
    SET(EIGEN3_FUTURE FALSE)
  ENDIF(${EIGEN3_VERSION} VERSION_GREATER "3.2.10")
ENDIF(EIGEN3_FOUND)

SET(BOOST_REQUIERED_COMPONENTS filesystem system)
SET(BOOST_BUILD_COMPONENTS unit_test_framework)
SET(BOOST_OPTIONAL_COMPONENTS "")

IF(BUILD_PYTHON_INTERFACE)
  SET(BOOST_OPTIONAL_COMPONENTS ${BOOST_OPTIONAL_COMPONENTS} python)
  FINDPYTHON(2.7 EXACT REQUIRED)
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS}) 
ENDIF(BUILD_PYTHON_INTERFACE)

SET(BOOST_COMPONENTS ${BOOST_REQUIERED_COMPONENTS} ${BOOST_OPTIONAL_COMPONENTS} ${BOOST_BUILD_COMPONENTS})
SEARCH_FOR_BOOST()

# Variable containing all the cflags definition relative to optional dependencies
SET(CFLAGS_DEPENDENCIES)

# Path to boost headers
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------

SET(HEADERS
  pycado.hpp
  )

MAKE_DIRECTORY("${${PROJECT_NAME}_BINARY_DIR}/include/pycado")

FOREACH(header ${HEADERS})
  GET_FILENAME_COMPONENT(headerName ${header} NAME)
  GET_FILENAME_COMPONENT(headerPath ${header} PATH)
  EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E ${LINK}
    ${${PROJECT_NAME}_SOURCE_DIR}/src/${header}
    ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/${header})
  INSTALL(FILES ${${PROJECT_NAME}_SOURCE_DIR}/src/${header}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${headerPath}
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
ENDFOREACH(header)

# --- MAIN LIBRARY -------------------------------------------------------------
ADD_SUBDIRECTORY(src)

# --- BINDINGS -----------------------------------------------------------------
ADD_SUBDIRECTORY(bindings)

# --- UNIT TESTS ---------------------------------------------------------------
ADD_SUBDIRECTORY(unittest)

# --- PACKAGING ----------------------------------------------------------------
PKG_CONFIG_APPEND_LIBS (${PROJECT_NAME})
PKG_CONFIG_APPEND_BOOST_LIBS(${BOOST_REQUIERED_COMPONENTS})

FOREACH(cflags ${CFLAGS_DEPENDENCIES})
  PKG_CONFIG_APPEND_CFLAGS(${cflags})
ENDFOREACH(cflags ${CFLAGS_DEPENDENCIES})

SETUP_PROJECT_FINALIZE()
