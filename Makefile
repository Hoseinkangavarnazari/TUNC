# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/hoseinknazari/Desktop/TUNC/CMakeFiles /home/hoseinknazari/Desktop/TUNC//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/hoseinknazari/Desktop/TUNC/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named runner.out

# Build rule for target.
runner.out: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 runner.out
.PHONY : runner.out

# fast build rule for target.
runner.out/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/build
.PHONY : runner.out/fast

headers/src/cFunctions.o: headers/src/cFunctions.cpp.o
.PHONY : headers/src/cFunctions.o

# target to build an object file
headers/src/cFunctions.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.o
.PHONY : headers/src/cFunctions.cpp.o

headers/src/cFunctions.i: headers/src/cFunctions.cpp.i
.PHONY : headers/src/cFunctions.i

# target to preprocess a source file
headers/src/cFunctions.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.i
.PHONY : headers/src/cFunctions.cpp.i

headers/src/cFunctions.s: headers/src/cFunctions.cpp.s
.PHONY : headers/src/cFunctions.s

# target to generate assembly for a file
headers/src/cFunctions.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/cFunctions.cpp.s
.PHONY : headers/src/cFunctions.cpp.s

headers/src/ff.o: headers/src/ff.cpp.o
.PHONY : headers/src/ff.o

# target to build an object file
headers/src/ff.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/ff.cpp.o
.PHONY : headers/src/ff.cpp.o

headers/src/ff.i: headers/src/ff.cpp.i
.PHONY : headers/src/ff.i

# target to preprocess a source file
headers/src/ff.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/ff.cpp.i
.PHONY : headers/src/ff.cpp.i

headers/src/ff.s: headers/src/ff.cpp.s
.PHONY : headers/src/ff.s

# target to generate assembly for a file
headers/src/ff.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/ff.cpp.s
.PHONY : headers/src/ff.cpp.s

headers/src/packet.o: headers/src/packet.cpp.o
.PHONY : headers/src/packet.o

# target to build an object file
headers/src/packet.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/packet.cpp.o
.PHONY : headers/src/packet.cpp.o

headers/src/packet.i: headers/src/packet.cpp.i
.PHONY : headers/src/packet.i

# target to preprocess a source file
headers/src/packet.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/packet.cpp.i
.PHONY : headers/src/packet.cpp.i

headers/src/packet.s: headers/src/packet.cpp.s
.PHONY : headers/src/packet.s

# target to generate assembly for a file
headers/src/packet.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/packet.cpp.s
.PHONY : headers/src/packet.cpp.s

headers/src/rlnc_decoder.o: headers/src/rlnc_decoder.cpp.o
.PHONY : headers/src/rlnc_decoder.o

# target to build an object file
headers/src/rlnc_decoder.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.o
.PHONY : headers/src/rlnc_decoder.cpp.o

headers/src/rlnc_decoder.i: headers/src/rlnc_decoder.cpp.i
.PHONY : headers/src/rlnc_decoder.i

# target to preprocess a source file
headers/src/rlnc_decoder.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.i
.PHONY : headers/src/rlnc_decoder.cpp.i

headers/src/rlnc_decoder.s: headers/src/rlnc_decoder.cpp.s
.PHONY : headers/src/rlnc_decoder.s

# target to generate assembly for a file
headers/src/rlnc_decoder.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/rlnc_decoder.cpp.s
.PHONY : headers/src/rlnc_decoder.cpp.s

headers/src/rlnc_encoder.o: headers/src/rlnc_encoder.cpp.o
.PHONY : headers/src/rlnc_encoder.o

# target to build an object file
headers/src/rlnc_encoder.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.o
.PHONY : headers/src/rlnc_encoder.cpp.o

headers/src/rlnc_encoder.i: headers/src/rlnc_encoder.cpp.i
.PHONY : headers/src/rlnc_encoder.i

# target to preprocess a source file
headers/src/rlnc_encoder.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.i
.PHONY : headers/src/rlnc_encoder.cpp.i

headers/src/rlnc_encoder.s: headers/src/rlnc_encoder.cpp.s
.PHONY : headers/src/rlnc_encoder.s

# target to generate assembly for a file
headers/src/rlnc_encoder.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/headers/src/rlnc_encoder.cpp.s
.PHONY : headers/src/rlnc_encoder.cpp.s

main.o: main.cpp.o
.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i
.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s
.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/runner.out.dir/build.make CMakeFiles/runner.out.dir/main.cpp.s
.PHONY : main.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... runner.out"
	@echo "... headers/src/cFunctions.o"
	@echo "... headers/src/cFunctions.i"
	@echo "... headers/src/cFunctions.s"
	@echo "... headers/src/ff.o"
	@echo "... headers/src/ff.i"
	@echo "... headers/src/ff.s"
	@echo "... headers/src/packet.o"
	@echo "... headers/src/packet.i"
	@echo "... headers/src/packet.s"
	@echo "... headers/src/rlnc_decoder.o"
	@echo "... headers/src/rlnc_decoder.i"
	@echo "... headers/src/rlnc_decoder.s"
	@echo "... headers/src/rlnc_encoder.o"
	@echo "... headers/src/rlnc_encoder.i"
	@echo "... headers/src/rlnc_encoder.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

