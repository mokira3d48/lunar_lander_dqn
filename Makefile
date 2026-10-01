# Installation of the Dueling DQN project in one command:
#
#   make install        detects the hardware and installs the right PyTorch build
#
# or force a build:
#
#   make cpu_install    CPU only (no GPU, or to save disk space)
#   make cuda_install   NVIDIA GPU (CUDA)
#   make rocm_install   AMD GPU (ROCm, Linux only)
#
# Each target creates the virtual environment `.venv` (if missing), installs
# PyTorch from the matching PyTorch index, then the other dependencies from
# requirements.txt. requirements.txt pins `torch==2.8.0`, which the build
# installed first already satisfies (e.g. 2.8.0+cpu), so it is not replaced
# by the default PyPI build.
#
# uv is used when available (faster), otherwise python -m venv + pip.
#
# Variables you can override on the command line:
#   make cuda_install CUDA=cu128     another CUDA version (cu126, cu128, cu129)
#   make rocm_install ROCM=rocm6.3   another ROCm version (rocm6.3, rocm6.4)
#   make install PYTHON=3.12         Python of the venv (with uv: a version,
#                                    downloaded if missing; without uv: a
#                                    command such as python3.12)
#   make install VENV=path/to/venv   location of the virtual environment
#
# PyTorch 2.8.0 has wheels for Python 3.9 to 3.13 only: a venv created with
# a newer Python (e.g. 3.14) cannot install it, and the install stops with
# an explicit message.

TORCH_VERSION ?= 2.8.0
CUDA ?= cu126
ROCM ?= rocm6.4
VENV ?= .venv
TORCH_INDEX = https://download.pytorch.org/whl

VENV_PYTHON = $(VENV)/bin/python
UV := $(shell command -v uv 2>/dev/null)
OS := $(shell uname -s)
ARCH := $(shell uname -m)

# Box2D, the physics engine of LunarLander. `gymnasium[box2d]` (in
# requirements.txt) installs the `box2d` package, which only exists as
# prebuilt wheels for Linux x86_64/i686, macOS and Windows: it has no source
# package, so it cannot be installed anywhere else (Linux ARM: Raspberry Pi,
# Jetson, ARM cloud servers...). There, `box2d-py` is used instead: it
# provides the same `Box2D` module and is compiled from source, which needs
# SWIG, a C++ compiler and the Python headers (see `box2d_check`).
# BOX2D_BUILD=yes forces this source build on any machine.
ifeq ($(OS),Linux)
  ifeq ($(filter $(ARCH),x86_64 i686 i386),)
    BOX2D_BUILD ?= yes
  endif
endif
BOX2D_BUILD ?= no

ifdef UV
  PIP = uv pip install --python $(VENV_PYTHON)
  PYTHON ?= 3.10
else
  PIP = $(VENV_PYTHON) -m pip install
  # Newest supported interpreter found on the PATH.
  PYTHON ?= $(shell for v in 3.13 3.12 3.11 3.10; do \
    command -v python$$v >/dev/null 2>&1 && { echo python$$v; break; }; done)
endif

# Install requirements.txt.
ifeq ($(BOX2D_BUILD),yes)
  # 1. Compile box2d-py with the SWIG, setuptools and wheel installed in the
  #    venv by `box2d_check`: no build isolation, and the venv's bin/ first
  #    on the PATH so that the build runs that `swig`. (In pip's isolated
  #    build environment, the venv's `swig` command cannot import its own
  #    Python module and the build fails.)
  # 2. The rest of requirements.txt, with plain `gymnasium` instead of
  #    `gymnasium[box2d]`, plus pygame-ce (the window), which the box2d
  #    extra would bring.
  BOX2D_REQUIREMENTS = $(VENV)/requirements-box2d-source.txt
  INSTALL_REQUIREMENTS = \
    PATH="$(abspath $(VENV))/bin:$$PATH" $(PIP) --no-build-isolation "box2d-py==2.3.5" && \
    sed 's/^gymnasium\[box2d\]/gymnasium/' requirements.txt > $(BOX2D_REQUIREMENTS) && \
    $(PIP) -r $(BOX2D_REQUIREMENTS) "pygame-ce>=2.1.3"
else
  INSTALL_REQUIREMENTS = $(PIP) -r requirements.txt
endif

# Hardware detection used by `make install`:
#   - NVIDIA GPU: `nvidia-smi -L` lists at least one GPU   -> cuda
#   - AMD GPU:    `rocminfo` reports a gfx* agent          -> rocm
#   - otherwise (including macOS)                          -> cpu
ACCEL := $(shell \
  if command -v nvidia-smi >/dev/null 2>&1 \
     && nvidia-smi -L 2>/dev/null | grep -q GPU; then echo cuda; \
  elif command -v rocminfo >/dev/null 2>&1 \
     && rocminfo 2>/dev/null | grep -q gfx; then echo rocm; \
  else echo cpu; fi)

.PHONY: install cpu_install cuda_install rocm_install python_check box2d_check \
  check   train run clean help

help:
	@echo "make install       detect the hardware ($(ACCEL) here) and install everything"
	@echo "make cpu_install   install with CPU-only PyTorch"
	@echo "make cuda_install  install with CUDA PyTorch ($(CUDA))"
	@echo "make rocm_install  install with ROCm PyTorch ($(ROCM))"
	@echo "make check         check that PyTorch and LunarLander work"
	@echo "make train         train the agent"
	@echo "make run DIR=...   play with the checkpoint in DIR (default: tmp/dqn)"
	@echo "make clean         remove Python caches"

install: $(ACCEL)_install

# Create the virtual environment only if it does not exist yet.
$(VENV_PYTHON):
ifdef UV
	uv venv -p $(PYTHON) $(VENV)
else
	@test -n "$(PYTHON)" || { echo "No python3.10 to python3.13 found:" \
	  "install one, or install uv (https://docs.astral.sh/uv/)."; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
endif

# Stop early if the venv's Python is not supported by PyTorch $(TORCH_VERSION).
python_check: $(VENV_PYTHON)
	@$(VENV_PYTHON) -c "import sys; v = sys.version_info[:2]; \
	  sys.exit(0 if (3, 9) <= v <= (3, 13) else \
	  'Python %d.%d in $(VENV) is not supported by PyTorch $(TORCH_VERSION) ' \
	  '(3.9 to 3.13). Remove $(VENV) and run make again.' % v)"

# When Box2D must be compiled: stop early if the C++ compiler or the Python
# headers are missing (they need the system package manager), and install
# SWIG and the build tools in the venv (no root rights needed).
box2d_check: python_check
ifeq ($(BOX2D_BUILD),yes)
	@echo "No prebuilt Box2D for $(OS) $(ARCH): box2d-py will be compiled from source."
	@command -v c++ >/dev/null 2>&1 || { echo "No C++ compiler found. Install one," \
	  "e.g. on Debian/Ubuntu: sudo apt install build-essential"; exit 1; }
	@$(VENV_PYTHON) -c "import os, sys, sysconfig; \
	  h = os.path.join(sysconfig.get_paths()['include'], 'Python.h'); \
	  sys.exit(0 if os.path.isfile(h) else \
	  'Python headers not found (%s). Install them, e.g. on Debian/Ubuntu: ' \
	  'sudo apt install python%d.%d-dev' % ((h,) + sys.version_info[:2]))"
	$(PIP) "swig==4.*" setuptools wheel
endif

# On macOS the default PyPI build is the right one (CPU + Apple MPS); the
# PyTorch CPU index is for Linux and Windows.
cpu_install: box2d_check
ifeq ($(OS),Darwin)
	$(PIP) torch==$(TORCH_VERSION)
else
	$(PIP) torch==$(TORCH_VERSION) --index-url $(TORCH_INDEX)/cpu
endif
	$(INSTALL_REQUIREMENTS)
	@$(MAKE) --no-print-directory check

cuda_install: box2d_check
	$(PIP) torch==$(TORCH_VERSION) --index-url $(TORCH_INDEX)/$(CUDA)
	$(INSTALL_REQUIREMENTS)
	@$(MAKE) --no-print-directory check

rocm_install: box2d_check
	$(PIP) torch==$(TORCH_VERSION) --index-url $(TORCH_INDEX)/$(ROCM)
	$(INSTALL_REQUIREMENTS)
	@$(MAKE) --no-print-directory check

# Print the installed PyTorch build, whether it sees a GPU, and the first
# state of LunarLander (fails if Box2D is missing).
check:
	$(VENV_PYTHON) -c "import torch; print('torch', torch.__version__, '- GPU available:', torch.cuda.is_available())"
	$(VENV_PYTHON) -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); print('LunarLander OK, initial state:', env.reset(seed=0)[0])"

train:
	$(VENV_PYTHON) dualing_dqn_torch.py

DIR ?= tmp/dqn
run:
	$(VENV_PYTHON) run_dualing_dqn_torch.py $(DIR)

clean:
	rm -rf __pycache__
