# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/linyihong/findcount/presolve

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/linyihong/findcount/presolve/build

# Include any dependencies generated for this target.
include CMakeFiles/presolve.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/presolve.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/presolve.dir/flags.make

CMakeFiles/presolve.dir/main.cpp.o: CMakeFiles/presolve.dir/flags.make
CMakeFiles/presolve.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/linyihong/findcount/presolve/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/presolve.dir/main.cpp.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/presolve.dir/main.cpp.o -c /home/linyihong/findcount/presolve/main.cpp

CMakeFiles/presolve.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/presolve.dir/main.cpp.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/linyihong/findcount/presolve/main.cpp > CMakeFiles/presolve.dir/main.cpp.i

CMakeFiles/presolve.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/presolve.dir/main.cpp.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/linyihong/findcount/presolve/main.cpp -o CMakeFiles/presolve.dir/main.cpp.s

CMakeFiles/presolve.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/presolve.dir/main.cpp.o.requires

CMakeFiles/presolve.dir/main.cpp.o.provides: CMakeFiles/presolve.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/presolve.dir/build.make CMakeFiles/presolve.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/presolve.dir/main.cpp.o.provides

CMakeFiles/presolve.dir/main.cpp.o.provides.build: CMakeFiles/presolve.dir/main.cpp.o


# Object files for target presolve
presolve_OBJECTS = \
"CMakeFiles/presolve.dir/main.cpp.o"

# External object files for target presolve
presolve_EXTERNAL_OBJECTS =

presolve: CMakeFiles/presolve.dir/main.cpp.o
presolve: CMakeFiles/presolve.dir/build.make
presolve: /usr/local/lib/libopencv_dnn.so.4.0.0
presolve: /usr/local/lib/libopencv_gapi.so.4.0.0
presolve: /usr/local/lib/libopencv_ml.so.4.0.0
presolve: /usr/local/lib/libopencv_objdetect.so.4.0.0
presolve: /usr/local/lib/libopencv_photo.so.4.0.0
presolve: /usr/local/lib/libopencv_stitching.so.4.0.0
presolve: /usr/local/lib/libopencv_video.so.4.0.0
presolve: /usr/local/lib/libopencv_calib3d.so.4.0.0
presolve: /usr/local/lib/libopencv_features2d.so.4.0.0
presolve: /usr/local/lib/libopencv_flann.so.4.0.0
presolve: /usr/local/lib/libopencv_highgui.so.4.0.0
presolve: /usr/local/lib/libopencv_videoio.so.4.0.0
presolve: /usr/local/lib/libopencv_imgcodecs.so.4.0.0
presolve: /usr/local/lib/libopencv_imgproc.so.4.0.0
presolve: /usr/local/lib/libopencv_core.so.4.0.0
presolve: CMakeFiles/presolve.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/linyihong/findcount/presolve/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable presolve"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/presolve.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/presolve.dir/build: presolve

.PHONY : CMakeFiles/presolve.dir/build

CMakeFiles/presolve.dir/requires: CMakeFiles/presolve.dir/main.cpp.o.requires

.PHONY : CMakeFiles/presolve.dir/requires

CMakeFiles/presolve.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/presolve.dir/cmake_clean.cmake
.PHONY : CMakeFiles/presolve.dir/clean

CMakeFiles/presolve.dir/depend:
	cd /home/linyihong/findcount/presolve/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/linyihong/findcount/presolve /home/linyihong/findcount/presolve /home/linyihong/findcount/presolve/build /home/linyihong/findcount/presolve/build /home/linyihong/findcount/presolve/build/CMakeFiles/presolve.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/presolve.dir/depend

