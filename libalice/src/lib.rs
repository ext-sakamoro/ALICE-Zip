//! libalice - High-Performance Procedural Generation Library
//!
//! Native Rust implementation of ALICE-Zip generators for maximum performance.
//! Provides Python bindings via PyO3 (optional).
//!
//! # Features
//! - Vectorized Perlin noise generation (parallel with Rayon)
//! - FFT-based Fourier signal reconstruction
//! - Polynomial evaluation with Horner's method
//! - LZMA/zlib compression with quantization
//!
//! # License
//! MIT License
//!
//! # Author
//! Moroya Sakamoto

pub mod generators;
pub mod compression;
pub mod ffi;

// ============================================================================
// Python Bindings (only when python feature is enabled)
// ============================================================================

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;
    use numpy::{PyArray1, PyArray2, IntoPyArray, PyArrayMethods};
    use super::{generators, compression};

    /// libalice - High-performance procedural generation for ALICE-Zip
    #[pymodule]
    pub fn libalice(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Generation functions
        m.add_function(wrap_pyfunction!(perlin_2d, m)?)?;
        m.add_function(wrap_pyfunction!(perlin_advanced, m)?)?;
        m.add_function(wrap_pyfunction!(fourier_generate, m)?)?;
        m.add_function(wrap_pyfunction!(sine_wave, m)?)?;
        m.add_function(wrap_pyfunction!(multi_sine, m)?)?;
        m.add_function(wrap_pyfunction!(polynomial_generate, m)?)?;
        m.add_function(wrap_pyfunction!(polynomial_fit, m)?)?;
        m.add_function(wrap_pyfunction!(fourier_analyze, m)?)?;

        // Compression functions
        m.add_function(wrap_pyfunction!(residual_compress, m)?)?;
        m.add_function(wrap_pyfunction!(residual_decompress, m)?)?;
        m.add_function(wrap_pyfunction!(residual_compress_lossless, m)?)?;
        m.add_function(wrap_pyfunction!(residual_decompress_lossless, m)?)?;
        m.add_function(wrap_pyfunction!(lzma_compress, m)?)?;
        m.add_function(wrap_pyfunction!(lzma_decompress, m)?)?;
        m.add_function(wrap_pyfunction!(zlib_compress, m)?)?;
        m.add_function(wrap_pyfunction!(zlib_decompress, m)?)?;
        Ok(())
    }

    // ============================================================================
    // Perlin Noise Functions
    // ============================================================================

    #[pyfunction]
    #[pyo3(signature = (width, height, seed=42, scale=10.0, octaves=4))]
    fn perlin_2d<'py>(
        py: Python<'py>,
        width: usize,
        height: usize,
        seed: u64,
        scale: f32,
        octaves: u32,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        // Validate input parameters
        if width == 0 || height == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "width and height must be greater than 0"
            ));
        }

        let data = generators::generate_perlin_2d(width, height, seed, scale, octaves);
        let array: Vec<Vec<f32>> = data
            .chunks(width)
            .map(|row| row.to_vec())
            .collect();
        let flat: Vec<f32> = array.into_iter().flatten().collect();

        numpy::PyArray::from_vec(py, flat)
            .reshape([height, width])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to reshape array: {:?}", e)
            ))
    }

    #[pyfunction]
    #[pyo3(signature = (width, height, seed=42, scale=10.0, octaves=4, persistence=0.5, lacunarity=2.0))]
    fn perlin_advanced<'py>(
        py: Python<'py>,
        width: usize,
        height: usize,
        seed: u64,
        scale: f32,
        octaves: u32,
        persistence: f32,
        lacunarity: f32,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        // Validate input parameters
        if width == 0 || height == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "width and height must be greater than 0"
            ));
        }
        if scale <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "scale must be greater than 0"
            ));
        }
        if octaves == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "octaves must be greater than 0"
            ));
        }

        let data = generators::generate_perlin_advanced(
            width, height, seed, scale, octaves, persistence, lacunarity
        );
        let flat: Vec<f32> = data;

        numpy::PyArray::from_vec(py, flat)
            .reshape([height, width])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to reshape array: {:?}", e)
            ))
    }

    // ============================================================================
    // Fourier Functions
    // ============================================================================

    #[pyfunction]
    #[pyo3(signature = (n, coefficients, dc_offset=0.0))]
    fn fourier_generate<'py>(
        py: Python<'py>,
        n: usize,
        coefficients: Vec<(usize, f32, f32)>,
        dc_offset: f32,
    ) -> Bound<'py, PyArray1<f32>> {
        let data = generators::generate_from_coefficients(n, &coefficients, dc_offset);
        data.into_pyarray(py)
    }

    #[pyfunction]
    #[pyo3(signature = (n, frequency, amplitude=1.0, phase=0.0, dc_offset=0.0))]
    fn sine_wave<'py>(
        py: Python<'py>,
        n: usize,
        frequency: f32,
        amplitude: f32,
        phase: f32,
        dc_offset: f32,
    ) -> Bound<'py, PyArray1<f32>> {
        let data = generators::generate_sine_wave(n, frequency, amplitude, phase, dc_offset);
        data.into_pyarray(py)
    }

    #[pyfunction]
    #[pyo3(signature = (n, components, dc_offset=0.0))]
    fn multi_sine<'py>(
        py: Python<'py>,
        n: usize,
        components: Vec<(f32, f32, f32)>,
        dc_offset: f32,
    ) -> Bound<'py, PyArray1<f32>> {
        let data = generators::generate_multi_sine(n, &components, dc_offset);
        data.into_pyarray(py)
    }

    #[pyfunction]
    #[pyo3(signature = (signal, max_coefficients=20, energy_threshold=0.99))]
    fn fourier_analyze(
        signal: Vec<f32>,
        max_coefficients: usize,
        energy_threshold: f32,
    ) -> (Vec<(usize, f32, f32)>, f32) {
        generators::analyze_signal(&signal, max_coefficients, energy_threshold)
    }

    // ============================================================================
    // Polynomial Functions
    // ============================================================================

    #[pyfunction]
    fn polynomial_generate<'py>(
        py: Python<'py>,
        n: usize,
        coefficients: Vec<f64>,
    ) -> Bound<'py, PyArray1<f32>> {
        let data = generators::generate_polynomial(n, &coefficients);
        data.into_pyarray(py)
    }

    #[pyfunction]
    #[pyo3(signature = (data, max_degree=9, error_threshold=0.001))]
    fn polynomial_fit(
        data: Vec<f32>,
        max_degree: usize,
        error_threshold: f64,
    ) -> Option<(Vec<f64>, usize, f64)> {
        generators::fit_polynomial(&data, max_degree, error_threshold)
    }

    // ============================================================================
    // Compression Functions
    // ============================================================================

    /// Convert std::io::Error to PyErr
    fn io_err_to_pyerr(e: std::io::Error) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Compression error: {}", e))
    }

    #[pyfunction]
    #[pyo3(signature = (residual, bits=8, lzma_preset=6))]
    fn residual_compress(
        residual: Vec<f32>,
        bits: u8,
        lzma_preset: u32,
    ) -> PyResult<Vec<u8>> {
        compression::compress_residual_quantized(&residual, bits, lzma_preset)
            .map_err(io_err_to_pyerr)
    }

    #[pyfunction]
    fn residual_decompress(data: Vec<u8>) -> PyResult<Vec<f32>> {
        compression::decompress_residual_quantized(&data)
            .map_err(io_err_to_pyerr)
    }

    #[pyfunction]
    #[pyo3(signature = (residual, lzma_preset=6))]
    fn residual_compress_lossless(
        residual: Vec<f32>,
        lzma_preset: u32,
    ) -> PyResult<Vec<u8>> {
        compression::compress_residual_lossless(&residual, lzma_preset)
            .map_err(io_err_to_pyerr)
    }

    #[pyfunction]
    fn residual_decompress_lossless(data: Vec<u8>) -> PyResult<Vec<f32>> {
        compression::decompress_residual_lossless(&data)
            .map_err(io_err_to_pyerr)
    }

    #[pyfunction]
    #[pyo3(signature = (data, preset=6))]
    fn lzma_compress(data: Vec<u8>, preset: u32) -> PyResult<Vec<u8>> {
        compression::lzma_compress(&data, preset)
            .map_err(io_err_to_pyerr)
    }

    #[pyfunction]
    fn lzma_decompress(data: Vec<u8>) -> PyResult<Vec<u8>> {
        compression::lzma_decompress(&data)
            .map_err(io_err_to_pyerr)
    }

    #[pyfunction]
    #[pyo3(signature = (data, level=6))]
    fn zlib_compress(data: Vec<u8>, level: u32) -> PyResult<Vec<u8>> {
        compression::zlib_compress(&data, level)
            .map_err(io_err_to_pyerr)
    }

    #[pyfunction]
    fn zlib_decompress(data: Vec<u8>) -> PyResult<Vec<u8>> {
        compression::zlib_decompress(&data)
            .map_err(io_err_to_pyerr)
    }
}

// Re-export the Python module when python feature is enabled
#[cfg(feature = "python")]
pub use python::libalice;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generators_module() {
        // Test Perlin noise
        let noise = generators::generate_perlin_2d(64, 64, 42, 10.0, 4);
        assert_eq!(noise.len(), 64 * 64);

        // Test sine wave
        let sine = generators::generate_sine_wave(100, 5.0, 1.0, 0.0, 0.0);
        assert_eq!(sine.len(), 100);

        // Test polynomial
        let poly = generators::generate_polynomial(100, &[1.0, 0.0, 0.0]);
        assert_eq!(poly.len(), 100);
    }

    #[test]
    fn test_compression_module() {
        // Test LZMA roundtrip
        let data = b"Hello, World! This is a test.".to_vec();
        let compressed = compression::lzma_compress(&data, 6).unwrap();
        let decompressed = compression::lzma_decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);

        // Test quantization roundtrip
        let floats: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let (quantized, min_val, scale) = compression::quantize_8bit(&floats);
        let restored = compression::dequantize_8bit(&quantized, min_val, scale);
        for (a, b) in floats.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.05);
        }
    }
}
