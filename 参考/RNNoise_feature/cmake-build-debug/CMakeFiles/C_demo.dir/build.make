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
CMAKE_SOURCE_DIR = /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/C_demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/C_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/C_demo.dir/flags.make

CMakeFiles/C_demo.dir/main.c.o: CMakeFiles/C_demo.dir/flags.make
CMakeFiles/C_demo.dir/main.c.o: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/C_demo.dir/main.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/C_demo.dir/main.c.o   -c /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/main.c

CMakeFiles/C_demo.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/C_demo.dir/main.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/main.c > CMakeFiles/C_demo.dir/main.c.i

CMakeFiles/C_demo.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/C_demo.dir/main.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/main.c -o CMakeFiles/C_demo.dir/main.c.s

CMakeFiles/C_demo.dir/main.c.o.requires:

.PHONY : CMakeFiles/C_demo.dir/main.c.o.requires

CMakeFiles/C_demo.dir/main.c.o.provides: CMakeFiles/C_demo.dir/main.c.o.requires
	$(MAKE) -f CMakeFiles/C_demo.dir/build.make CMakeFiles/C_demo.dir/main.c.o.provides.build
.PHONY : CMakeFiles/C_demo.dir/main.c.o.provides

CMakeFiles/C_demo.dir/main.c.o.provides.build: CMakeFiles/C_demo.dir/main.c.o


CMakeFiles/C_demo.dir/src/celt_lpc.c.o: CMakeFiles/C_demo.dir/flags.make
CMakeFiles/C_demo.dir/src/celt_lpc.c.o: ../src/celt_lpc.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/C_demo.dir/src/celt_lpc.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/C_demo.dir/src/celt_lpc.c.o   -c /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/celt_lpc.c

CMakeFiles/C_demo.dir/src/celt_lpc.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/C_demo.dir/src/celt_lpc.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/celt_lpc.c > CMakeFiles/C_demo.dir/src/celt_lpc.c.i

CMakeFiles/C_demo.dir/src/celt_lpc.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/C_demo.dir/src/celt_lpc.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/celt_lpc.c -o CMakeFiles/C_demo.dir/src/celt_lpc.c.s

CMakeFiles/C_demo.dir/src/celt_lpc.c.o.requires:

.PHONY : CMakeFiles/C_demo.dir/src/celt_lpc.c.o.requires

CMakeFiles/C_demo.dir/src/celt_lpc.c.o.provides: CMakeFiles/C_demo.dir/src/celt_lpc.c.o.requires
	$(MAKE) -f CMakeFiles/C_demo.dir/build.make CMakeFiles/C_demo.dir/src/celt_lpc.c.o.provides.build
.PHONY : CMakeFiles/C_demo.dir/src/celt_lpc.c.o.provides

CMakeFiles/C_demo.dir/src/celt_lpc.c.o.provides.build: CMakeFiles/C_demo.dir/src/celt_lpc.c.o


CMakeFiles/C_demo.dir/src/function.c.o: CMakeFiles/C_demo.dir/flags.make
CMakeFiles/C_demo.dir/src/function.c.o: ../src/function.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/C_demo.dir/src/function.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/C_demo.dir/src/function.c.o   -c /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/function.c

CMakeFiles/C_demo.dir/src/function.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/C_demo.dir/src/function.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/function.c > CMakeFiles/C_demo.dir/src/function.c.i

CMakeFiles/C_demo.dir/src/function.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/C_demo.dir/src/function.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/function.c -o CMakeFiles/C_demo.dir/src/function.c.s

CMakeFiles/C_demo.dir/src/function.c.o.requires:

.PHONY : CMakeFiles/C_demo.dir/src/function.c.o.requires

CMakeFiles/C_demo.dir/src/function.c.o.provides: CMakeFiles/C_demo.dir/src/function.c.o.requires
	$(MAKE) -f CMakeFiles/C_demo.dir/build.make CMakeFiles/C_demo.dir/src/function.c.o.provides.build
.PHONY : CMakeFiles/C_demo.dir/src/function.c.o.provides

CMakeFiles/C_demo.dir/src/function.c.o.provides.build: CMakeFiles/C_demo.dir/src/function.c.o


CMakeFiles/C_demo.dir/src/kiss_fft.c.o: CMakeFiles/C_demo.dir/flags.make
CMakeFiles/C_demo.dir/src/kiss_fft.c.o: ../src/kiss_fft.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/C_demo.dir/src/kiss_fft.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/C_demo.dir/src/kiss_fft.c.o   -c /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/kiss_fft.c

CMakeFiles/C_demo.dir/src/kiss_fft.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/C_demo.dir/src/kiss_fft.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/kiss_fft.c > CMakeFiles/C_demo.dir/src/kiss_fft.c.i

CMakeFiles/C_demo.dir/src/kiss_fft.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/C_demo.dir/src/kiss_fft.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/kiss_fft.c -o CMakeFiles/C_demo.dir/src/kiss_fft.c.s

CMakeFiles/C_demo.dir/src/kiss_fft.c.o.requires:

.PHONY : CMakeFiles/C_demo.dir/src/kiss_fft.c.o.requires

CMakeFiles/C_demo.dir/src/kiss_fft.c.o.provides: CMakeFiles/C_demo.dir/src/kiss_fft.c.o.requires
	$(MAKE) -f CMakeFiles/C_demo.dir/build.make CMakeFiles/C_demo.dir/src/kiss_fft.c.o.provides.build
.PHONY : CMakeFiles/C_demo.dir/src/kiss_fft.c.o.provides

CMakeFiles/C_demo.dir/src/kiss_fft.c.o.provides.build: CMakeFiles/C_demo.dir/src/kiss_fft.c.o


CMakeFiles/C_demo.dir/src/pitch.c.o: CMakeFiles/C_demo.dir/flags.make
CMakeFiles/C_demo.dir/src/pitch.c.o: ../src/pitch.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/C_demo.dir/src/pitch.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/C_demo.dir/src/pitch.c.o   -c /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/pitch.c

CMakeFiles/C_demo.dir/src/pitch.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/C_demo.dir/src/pitch.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/pitch.c > CMakeFiles/C_demo.dir/src/pitch.c.i

CMakeFiles/C_demo.dir/src/pitch.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/C_demo.dir/src/pitch.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/pitch.c -o CMakeFiles/C_demo.dir/src/pitch.c.s

CMakeFiles/C_demo.dir/src/pitch.c.o.requires:

.PHONY : CMakeFiles/C_demo.dir/src/pitch.c.o.requires

CMakeFiles/C_demo.dir/src/pitch.c.o.provides: CMakeFiles/C_demo.dir/src/pitch.c.o.requires
	$(MAKE) -f CMakeFiles/C_demo.dir/build.make CMakeFiles/C_demo.dir/src/pitch.c.o.provides.build
.PHONY : CMakeFiles/C_demo.dir/src/pitch.c.o.provides

CMakeFiles/C_demo.dir/src/pitch.c.o.provides.build: CMakeFiles/C_demo.dir/src/pitch.c.o


CMakeFiles/C_demo.dir/src/rnn.c.o: CMakeFiles/C_demo.dir/flags.make
CMakeFiles/C_demo.dir/src/rnn.c.o: ../src/rnn.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/C_demo.dir/src/rnn.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/C_demo.dir/src/rnn.c.o   -c /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/rnn.c

CMakeFiles/C_demo.dir/src/rnn.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/C_demo.dir/src/rnn.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/rnn.c > CMakeFiles/C_demo.dir/src/rnn.c.i

CMakeFiles/C_demo.dir/src/rnn.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/C_demo.dir/src/rnn.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/rnn.c -o CMakeFiles/C_demo.dir/src/rnn.c.s

CMakeFiles/C_demo.dir/src/rnn.c.o.requires:

.PHONY : CMakeFiles/C_demo.dir/src/rnn.c.o.requires

CMakeFiles/C_demo.dir/src/rnn.c.o.provides: CMakeFiles/C_demo.dir/src/rnn.c.o.requires
	$(MAKE) -f CMakeFiles/C_demo.dir/build.make CMakeFiles/C_demo.dir/src/rnn.c.o.provides.build
.PHONY : CMakeFiles/C_demo.dir/src/rnn.c.o.provides

CMakeFiles/C_demo.dir/src/rnn.c.o.provides.build: CMakeFiles/C_demo.dir/src/rnn.c.o


CMakeFiles/C_demo.dir/src/rnn_data.c.o: CMakeFiles/C_demo.dir/flags.make
CMakeFiles/C_demo.dir/src/rnn_data.c.o: ../src/rnn_data.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/C_demo.dir/src/rnn_data.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/C_demo.dir/src/rnn_data.c.o   -c /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/rnn_data.c

CMakeFiles/C_demo.dir/src/rnn_data.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/C_demo.dir/src/rnn_data.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/rnn_data.c > CMakeFiles/C_demo.dir/src/rnn_data.c.i

CMakeFiles/C_demo.dir/src/rnn_data.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/C_demo.dir/src/rnn_data.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/rnn_data.c -o CMakeFiles/C_demo.dir/src/rnn_data.c.s

CMakeFiles/C_demo.dir/src/rnn_data.c.o.requires:

.PHONY : CMakeFiles/C_demo.dir/src/rnn_data.c.o.requires

CMakeFiles/C_demo.dir/src/rnn_data.c.o.provides: CMakeFiles/C_demo.dir/src/rnn_data.c.o.requires
	$(MAKE) -f CMakeFiles/C_demo.dir/build.make CMakeFiles/C_demo.dir/src/rnn_data.c.o.provides.build
.PHONY : CMakeFiles/C_demo.dir/src/rnn_data.c.o.provides

CMakeFiles/C_demo.dir/src/rnn_data.c.o.provides.build: CMakeFiles/C_demo.dir/src/rnn_data.c.o


CMakeFiles/C_demo.dir/src/rnn_reader.c.o: CMakeFiles/C_demo.dir/flags.make
CMakeFiles/C_demo.dir/src/rnn_reader.c.o: ../src/rnn_reader.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/C_demo.dir/src/rnn_reader.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/C_demo.dir/src/rnn_reader.c.o   -c /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/rnn_reader.c

CMakeFiles/C_demo.dir/src/rnn_reader.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/C_demo.dir/src/rnn_reader.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/rnn_reader.c > CMakeFiles/C_demo.dir/src/rnn_reader.c.i

CMakeFiles/C_demo.dir/src/rnn_reader.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/C_demo.dir/src/rnn_reader.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/src/rnn_reader.c -o CMakeFiles/C_demo.dir/src/rnn_reader.c.s

CMakeFiles/C_demo.dir/src/rnn_reader.c.o.requires:

.PHONY : CMakeFiles/C_demo.dir/src/rnn_reader.c.o.requires

CMakeFiles/C_demo.dir/src/rnn_reader.c.o.provides: CMakeFiles/C_demo.dir/src/rnn_reader.c.o.requires
	$(MAKE) -f CMakeFiles/C_demo.dir/build.make CMakeFiles/C_demo.dir/src/rnn_reader.c.o.provides.build
.PHONY : CMakeFiles/C_demo.dir/src/rnn_reader.c.o.provides

CMakeFiles/C_demo.dir/src/rnn_reader.c.o.provides.build: CMakeFiles/C_demo.dir/src/rnn_reader.c.o


# Object files for target C_demo
C_demo_OBJECTS = \
"CMakeFiles/C_demo.dir/main.c.o" \
"CMakeFiles/C_demo.dir/src/celt_lpc.c.o" \
"CMakeFiles/C_demo.dir/src/function.c.o" \
"CMakeFiles/C_demo.dir/src/kiss_fft.c.o" \
"CMakeFiles/C_demo.dir/src/pitch.c.o" \
"CMakeFiles/C_demo.dir/src/rnn.c.o" \
"CMakeFiles/C_demo.dir/src/rnn_data.c.o" \
"CMakeFiles/C_demo.dir/src/rnn_reader.c.o"

# External object files for target C_demo
C_demo_EXTERNAL_OBJECTS =

C_demo: CMakeFiles/C_demo.dir/main.c.o
C_demo: CMakeFiles/C_demo.dir/src/celt_lpc.c.o
C_demo: CMakeFiles/C_demo.dir/src/function.c.o
C_demo: CMakeFiles/C_demo.dir/src/kiss_fft.c.o
C_demo: CMakeFiles/C_demo.dir/src/pitch.c.o
C_demo: CMakeFiles/C_demo.dir/src/rnn.c.o
C_demo: CMakeFiles/C_demo.dir/src/rnn_data.c.o
C_demo: CMakeFiles/C_demo.dir/src/rnn_reader.c.o
C_demo: CMakeFiles/C_demo.dir/build.make
C_demo: CMakeFiles/C_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking C executable C_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/C_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/C_demo.dir/build: C_demo

.PHONY : CMakeFiles/C_demo.dir/build

CMakeFiles/C_demo.dir/requires: CMakeFiles/C_demo.dir/main.c.o.requires
CMakeFiles/C_demo.dir/requires: CMakeFiles/C_demo.dir/src/celt_lpc.c.o.requires
CMakeFiles/C_demo.dir/requires: CMakeFiles/C_demo.dir/src/function.c.o.requires
CMakeFiles/C_demo.dir/requires: CMakeFiles/C_demo.dir/src/kiss_fft.c.o.requires
CMakeFiles/C_demo.dir/requires: CMakeFiles/C_demo.dir/src/pitch.c.o.requires
CMakeFiles/C_demo.dir/requires: CMakeFiles/C_demo.dir/src/rnn.c.o.requires
CMakeFiles/C_demo.dir/requires: CMakeFiles/C_demo.dir/src/rnn_data.c.o.requires
CMakeFiles/C_demo.dir/requires: CMakeFiles/C_demo.dir/src/rnn_reader.c.o.requires

.PHONY : CMakeFiles/C_demo.dir/requires

CMakeFiles/C_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/C_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/C_demo.dir/clean

CMakeFiles/C_demo.dir/depend:
	cd /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug /mnt/c/Users/anker/Desktop/Perceptual_scale/参考/RNNoise_feature/cmake-build-debug/CMakeFiles/C_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/C_demo.dir/depend

