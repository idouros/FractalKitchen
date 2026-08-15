# FractalKitchen

FractalKitchen is an experimental fractal renderer written in modern C++ and OpenCL.

The project uses Mandelbrot and Julia sets as a test bed for exploring GPU computation, numerical precision, interactive rendering and colouring techniques.

## Features

- Mandelbrot and Julia set generation
- OpenCL-based parallel computation
- Double-double arithmetic for higher-precision complex calculations
- Configurable image bounds, iteration limits and rendering parameters
- Interactive navigation and re-rendering
- Multiple smooth colouring schemes
- CMake-based C++20 build

## Why fractals?

Fractal generation is a useful playground for numerical and high-performance computing.

The underlying recurrence is simple, but large images provide a naturally parallel workload, while deeper zoom levels quickly expose floating-point precision limitations. This makes the problem suitable for experimenting with GPU execution, numerical representation and rendering strategies without obscuring those issues behind a large application framework.

FractalKitchen is therefore intended primarily as an engineering and numerical-computing project rather than as a finished end-user fractal application.

## Implementation

The host application is written in C++20. OpenCL kernels perform the fractal computation, while OpenCV is used for image representation and display.

The current OpenCL implementation includes double-double arithmetic for complex-number operations in order to extend the useful numerical range beyond conventional floating-point calculations.

Runtime parameters are supplied through configuration files. Example configurations for both Mandelbrot and Julia rendering are included under `cpp/configs`.

## Repository structure

- `cpp/fkcl` — main C++ application and OpenCL kernels
- `cpp/configs` — example Mandelbrot and Julia configurations
- `toys` — small experimental/reference implementations

## Building

The project uses CMake and currently depends on:

- a C++20 compiler
- OpenCL
- OpenCV
- Boost

The current CMake configuration expects the OpenCL, OpenCV and Boost locations to be supplied through environment variables.

## Running

The executable expects a configuration file:

```text
fkcl <ConfigFile>
```

Example configuration files are provided in:

```text
cpp/configs/
```

For example:

```text
fkcl cpp/configs/mandelbrot01.config
```

or:

```text
fkcl cpp/configs/julia01.config
```

## Current status

FractalKitchen is an ongoing personal project and remains under active development.

Current areas of experimentation include numerical precision, GPU performance, interactive navigation and fractal colouring.
