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

# --- MACROS ------------------------------------------------------------------
# --- MACROS ------------------------------------------------------------------
# --- MACROS ------------------------------------------------------------------

MACRO(ADD_TEST_CFLAGS target flag)
  SET_PROPERTY(TARGET ${target} APPEND_STRING PROPERTY COMPILE_FLAGS " ${flag}")
ENDMACRO(ADD_TEST_CFLAGS)

MACRO(ADD_UNIT_TEST NAME PKGS)
  IF (BUILD_UNIT_TESTS)
    ADD_EXECUTABLE(${NAME} ${NAME})
  ELSE (BUILD_UNIT_TESTS)
    ADD_EXECUTABLE(${NAME} EXCLUDE_FROM_ALL ${NAME})
  ENDIF (BUILD_UNIT_TESTS)
  SET_TARGET_PROPERTIES(${NAME} PROPERTIES LINKER_LANGUAGE CXX)
  ADD_TEST_CFLAGS(${NAME} "-DBOOST_TEST_DYN_LINK") 
  SET(MODULE_NAME "${NAME}Test")
  STRING(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})
  ADD_TEST_CFLAGS(${NAME} "-DBOOST_TEST_MODULE=${MODULE_NAME}") 

  FOREACH(PKG ${PKGS})
    PKG_CONFIG_USE_DEPENDENCY(${NAME} ${PKG})
  ENDFOREACH(PKG)

  TARGET_LINK_LIBRARIES(${NAME} ${PROJECT_NAME})
  TARGET_LINK_LIBRARIES(${NAME} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  TARGET_LINK_LIBRARIES(${NAME} ${ACADO_SHARED_LIBRARIES})
  TARGET_LINK_LIBRARIES(${NAME} ${Boost_LIBRARIES} )

  ADD_TEST(NAME ${NAME} COMMAND ${NAME})
  ADD_DEPENDENCIES(check ${NAME})
ENDMACRO(ADD_UNIT_TEST)

# --- RULES -------------------------------------------------------------------
# --- RULES -------------------------------------------------------------------
# --- RULES -------------------------------------------------------------------
ADD_CUSTOM_TARGET(check COMMAND ${CMAKE_CTEST_COMMAND})

ADD_UNIT_TEST(pendulum eigen3)
ADD_UNIT_TEST(discrete_pendulum eigen3)
ADD_UNIT_TEST(discrete_double_pendulum eigen3)
ADD_UNIT_TEST(minitime eigen3)
ADD_UNIT_TEST(pendulum2o eigen3)
ADD_UNIT_TEST(connect_pendulum eigen3)
ADD_UNIT_TEST(connect_bicopter eigen3)
ADD_UNIT_TEST(connect_quadcopter eigen3)
ADD_UNIT_TEST(connect_double_pendulum eigen3)
