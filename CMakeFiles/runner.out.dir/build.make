# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hoseinknazari/Desktop/TUNC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hoseinknazari/Desktop/TUNC

# Include any dependencies generated for this target.
include CMakeFiles/runner.out.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/runner.out.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/runner.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/runner.out.dir/flags.make

CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o: CMakeFiles/runner.out.dir/flags.make
CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o: headers/src/cFunctions.cpp
CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o: CMakeFiles/runner.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hoseinknazari/Desktop/TUNC/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o -MF CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o.d -o CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o -c /home/hoseinknazari/Desktop/TUNC/headers/src/cFunctions.cpp

CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hoseinknazari/Desktop/TUNC/headers/src/cFunctions.cpp > CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.i

CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hoseinknazari/Desktop/TUNC/headers/src/cFunctions.cpp -o CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.s

CMakeFiles/runner.out.dir/headers/src/ff.cpp.o: CMakeFiles/runner.out.dir/flags.make
CMakeFiles/runner.out.dir/headers/src/ff.cpp.o: headers/src/ff.cpp
CMakeFiles/runner.out.dir/headers/src/ff.cpp.o: CMakeFiles/runner.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hoseinknazari/Desktop/TUNC/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/runner.out.dir/headers/src/ff.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/runner.out.dir/headers/src/ff.cpp.o -MF CMakeFiles/runner.out.dir/headers/src/ff.cpp.o.d -o CMakeFiles/runner.out.dir/headers/src/ff.cpp.o -c /home/hoseinknazari/Desktop/TUNC/headers/src/ff.cpp

CMakeFiles/runner.out.dir/headers/src/ff.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runner.out.dir/headers/src/ff.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hoseinknazari/Desktop/TUNC/headers/src/ff.cpp > CMakeFiles/runner.out.dir/headers/src/ff.cpp.i

CMakeFiles/runner.out.dir/headers/src/ff.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runner.out.dir/headers/src/ff.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hoseinknazari/Desktop/TUNC/headers/src/ff.cpp -o CMakeFiles/runner.out.dir/headers/src/ff.cpp.s

CMakeFiles/runner.out.dir/headers/src/packet.cpp.o: CMakeFiles/runner.out.dir/flags.make
CMakeFiles/runner.out.dir/headers/src/packet.cpp.o: headers/src/packet.cpp
CMakeFiles/runner.out.dir/headers/src/packet.cpp.o: CMakeFiles/runner.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hoseinknazari/Desktop/TUNC/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/runner.out.dir/headers/src/packet.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/runner.out.dir/headers/src/packet.cpp.o -MF CMakeFiles/runner.out.dir/headers/src/packet.cpp.o.d -o CMakeFiles/runner.out.dir/headers/src/packet.cpp.o -c /home/hoseinknazari/Desktop/TUNC/headers/src/packet.cpp

CMakeFiles/runner.out.dir/headers/src/packet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runner.out.dir/headers/src/packet.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hoseinknazari/Desktop/TUNC/headers/src/packet.cpp > CMakeFiles/runner.out.dir/headers/src/packet.cpp.i

CMakeFiles/runner.out.dir/headers/src/packet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runner.out.dir/headers/src/packet.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hoseinknazari/Desktop/TUNC/headers/src/packet.cpp -o CMakeFiles/runner.out.dir/headers/src/packet.cpp.s

CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o: CMakeFiles/runner.out.dir/flags.make
CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o: headers/src/rlnc_decoder.cpp
CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o: CMakeFiles/runner.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hoseinknazari/Desktop/TUNC/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o -MF CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o.d -o CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o -c /home/hoseinknazari/Desktop/TUNC/headers/src/rlnc_decoder.cpp

CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hoseinknazari/Desktop/TUNC/headers/src/rlnc_decoder.cpp > CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.i

CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hoseinknazari/Desktop/TUNC/headers/src/rlnc_decoder.cpp -o CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.s

CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o: CMakeFiles/runner.out.dir/flags.make
CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o: headers/src/rlnc_encoder.cpp
CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o: CMakeFiles/runner.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hoseinknazari/Desktop/TUNC/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o -MF CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o.d -o CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o -c /home/hoseinknazari/Desktop/TUNC/headers/src/rlnc_encoder.cpp

CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hoseinknazari/Desktop/TUNC/headers/src/rlnc_encoder.cpp > CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.i

CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hoseinknazari/Desktop/TUNC/headers/src/rlnc_encoder.cpp -o CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.s

CMakeFiles/runner.out.dir/main.cpp.o: CMakeFiles/runner.out.dir/flags.make
CMakeFiles/runner.out.dir/main.cpp.o: main.cpp
CMakeFiles/runner.out.dir/main.cpp.o: CMakeFiles/runner.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hoseinknazari/Desktop/TUNC/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/runner.out.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/runner.out.dir/main.cpp.o -MF CMakeFiles/runner.out.dir/main.cpp.o.d -o CMakeFiles/runner.out.dir/main.cpp.o -c /home/hoseinknazari/Desktop/TUNC/main.cpp

CMakeFiles/runner.out.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runner.out.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hoseinknazari/Desktop/TUNC/main.cpp > CMakeFiles/runner.out.dir/main.cpp.i

CMakeFiles/runner.out.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runner.out.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hoseinknazari/Desktop/TUNC/main.cpp -o CMakeFiles/runner.out.dir/main.cpp.s

# Object files for target runner.out
runner_out_OBJECTS = \
"CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o" \
"CMakeFiles/runner.out.dir/headers/src/ff.cpp.o" \
"CMakeFiles/runner.out.dir/headers/src/packet.cpp.o" \
"CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o" \
"CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o" \
"CMakeFiles/runner.out.dir/main.cpp.o"

# External object files for target runner.out
runner_out_EXTERNAL_OBJECTS =

runner.out: CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o
runner.out: CMakeFiles/runner.out.dir/headers/src/ff.cpp.o
runner.out: CMakeFiles/runner.out.dir/headers/src/packet.cpp.o
runner.out: CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o
runner.out: CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o
runner.out: CMakeFiles/runner.out.dir/main.cpp.o
runner.out: CMakeFiles/runner.out.dir/build.make
runner.out: CMakeFiles/runner.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hoseinknazari/Desktop/TUNC/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable runner.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/runner.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/runner.out.dir/build: runner.out
.PHONY : CMakeFiles/runner.out.dir/build

CMakeFiles/runner.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/runner.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/runner.out.dir/clean

CMakeFiles/runner.out.dir/depend:
	cd /home/hoseinknazari/Desktop/TUNC && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hoseinknazari/Desktop/TUNC /home/hoseinknazari/Desktop/TUNC /home/hoseinknazari/Desktop/TUNC /home/hoseinknazari/Desktop/TUNC /home/hoseinknazari/Desktop/TUNC/CMakeFiles/runner.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/runner.out.dir/depend

