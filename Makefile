.DEFAULT_GOAL := build

# Packages that can be built/tested without GPU toolchains (CUDA, ROCm).
NO_GPU_PKGS := -p inferrs -p inferrs-benchmark -p inferrs-backend-vulkan

.PHONY: all build release fmt clippy test

all: fmt clippy test build

build:
	cargo build $(NO_GPU_PKGS)

release:
	cargo build --release $(NO_GPU_PKGS)

fmt:
	cargo fmt --check $(NO_GPU_PKGS)

clippy:
	cargo clippy $(NO_GPU_PKGS) -- -D warnings

test:
	cargo test $(NO_GPU_PKGS)
