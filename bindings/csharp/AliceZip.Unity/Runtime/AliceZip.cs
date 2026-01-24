/**
 * ALICE-Zip C# Bindings
 *
 * High-performance procedural compression library.
 *
 * Author: Moroya Sakamoto
 * License: MIT (Core) / Game Industry Exception available
 *
 * Usage:
 *   // Generate Perlin noise texture
 *   float[] noise = AliceZip.PerlinNoise2D(256, 256, seed: 42);
 *
 *   // Compress data
 *   byte[] compressed = AliceZip.LzmaCompress(data);
 *   byte[] decompressed = AliceZip.LzmaDecompress(compressed);
 */

using System;
using System.Runtime.InteropServices;

namespace AliceZip
{
    /// <summary>
    /// Error codes returned by ALICE functions
    /// </summary>
    public enum AliceError
    {
        Success = 0,
        NullPointer = 1,
        InvalidParameter = 2,
        CompressionError = 3,
        DecompressionError = 4,
        AllocationError = 5,
        InvalidData = 6
    }

    /// <summary>
    /// Buffer structure for byte arrays
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct AliceBuffer
    {
        public IntPtr data;
        public UIntPtr len;
        public UIntPtr capacity;
    }

    /// <summary>
    /// Buffer structure for float arrays
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct AliceFloatBuffer
    {
        public IntPtr data;
        public UIntPtr len;
        public UIntPtr capacity;
    }

    /// <summary>
    /// Fourier coefficient for signal reconstruction
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FourierCoefficient
    {
        public UIntPtr frequency;
        public float amplitude;
        public float phase;

        public FourierCoefficient(int frequency, float amplitude, float phase)
        {
            this.frequency = (UIntPtr)frequency;
            this.amplitude = amplitude;
            this.phase = phase;
        }
    }

    /// <summary>
    /// ALICE-Zip native library bindings
    /// </summary>
    internal static class NativeMethods
    {
        #if UNITY_IOS && !UNITY_EDITOR
        private const string LibraryName = "__Internal";
        #elif UNITY_ANDROID && !UNITY_EDITOR
        private const string LibraryName = "alice_core";
        #elif UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        private const string LibraryName = "alice_core.dll";
        #elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        private const string LibraryName = "libalice_core.dylib";
        #elif UNITY_STANDALONE_LINUX || UNITY_EDITOR_LINUX
        private const string LibraryName = "libalice_core.so";
        #else
        private const string LibraryName = "alice_core";
        #endif

        // Memory management
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void alice_free_buffer(ref AliceBuffer buffer);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void alice_free_float_buffer(ref AliceFloatBuffer buffer);

        // Version info
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr alice_version();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void alice_version_numbers(out uint major, out uint minor, out uint patch);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr alice_get_last_error();

        // Perlin noise
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_perlin_2d(
            UIntPtr width, UIntPtr height, ulong seed,
            float scale, uint octaves, out AliceFloatBuffer buffer);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_perlin_advanced(
            UIntPtr width, UIntPtr height, ulong seed,
            float scale, uint octaves, float persistence, float lacunarity,
            out AliceFloatBuffer buffer);

        // Sine wave
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_sine_wave(
            UIntPtr n, float frequency, float amplitude, float phase, float dcOffset,
            out AliceFloatBuffer buffer);

        // Fourier
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_fourier_generate(
            UIntPtr n, [In] FourierCoefficient[] coefficients, UIntPtr numCoefficients,
            float dcOffset, out AliceFloatBuffer buffer);

        // Polynomial
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_polynomial_generate(
            UIntPtr n, [In] double[] coefficients, UIntPtr numCoefficients,
            out AliceFloatBuffer buffer);

        // LZMA compression
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_lzma_compress(
            [In] byte[] data, UIntPtr len, uint preset, out AliceBuffer buffer);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_lzma_decompress(
            [In] byte[] data, UIntPtr len, out AliceBuffer buffer);

        // zlib compression
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_zlib_compress(
            [In] byte[] data, UIntPtr len, uint level, out AliceBuffer buffer);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_zlib_decompress(
            [In] byte[] data, UIntPtr len, out AliceBuffer buffer);

        // Residual compression
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_residual_compress(
            [In] float[] residual, UIntPtr len, byte bits, uint lzmaPreset,
            out AliceBuffer buffer);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern AliceError alice_residual_decompress(
            [In] byte[] data, UIntPtr len, out AliceFloatBuffer buffer);
    }

    /// <summary>
    /// Exception thrown when an ALICE operation fails
    /// </summary>
    public class AliceException : Exception
    {
        public AliceError ErrorCode { get; }

        public AliceException(AliceError error, string message = null)
            : base(message ?? GetDefaultMessage(error))
        {
            ErrorCode = error;
        }

        private static string GetDefaultMessage(AliceError error)
        {
            return error switch
            {
                AliceError.NullPointer => "A required pointer was null",
                AliceError.InvalidParameter => "Invalid parameter value",
                AliceError.CompressionError => "Compression operation failed",
                AliceError.DecompressionError => "Decompression operation failed",
                AliceError.AllocationError => "Memory allocation failed",
                AliceError.InvalidData => "Input data is invalid or corrupted",
                _ => $"Unknown error: {error}"
            };
        }
    }

    /// <summary>
    /// Main ALICE-Zip API class
    /// </summary>
    public static class Alice
    {
        /// <summary>
        /// Get the native library version
        /// </summary>
        public static string Version
        {
            get
            {
                IntPtr ptr = NativeMethods.alice_version();
                return Marshal.PtrToStringAnsi(ptr);
            }
        }

        /// <summary>
        /// Get version numbers
        /// </summary>
        public static (uint Major, uint Minor, uint Patch) VersionNumbers
        {
            get
            {
                NativeMethods.alice_version_numbers(out uint major, out uint minor, out uint patch);
                return (major, minor, patch);
            }
        }

        /// <summary>
        /// Get the last error message
        /// </summary>
        public static string LastError
        {
            get
            {
                IntPtr ptr = NativeMethods.alice_get_last_error();
                return ptr == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(ptr);
            }
        }

        private static void CheckError(AliceError error)
        {
            if (error != AliceError.Success)
            {
                string message = LastError;
                throw new AliceException(error, message);
            }
        }

        #region Perlin Noise

        /// <summary>
        /// Generate 2D Perlin noise
        /// </summary>
        /// <param name="width">Width of the texture</param>
        /// <param name="height">Height of the texture</param>
        /// <param name="seed">Random seed</param>
        /// <param name="scale">Noise scale (larger = more zoomed out)</param>
        /// <param name="octaves">Number of octaves for fractal noise</param>
        /// <returns>Float array of noise values in row-major order</returns>
        public static float[] PerlinNoise2D(int width, int height, ulong seed = 42,
            float scale = 10.0f, uint octaves = 4)
        {
            AliceError error = NativeMethods.alice_perlin_2d(
                (UIntPtr)width, (UIntPtr)height, seed, scale, octaves, out AliceFloatBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                float[] result = new float[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_float_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Generate advanced 2D Perlin noise with persistence and lacunarity
        /// </summary>
        public static float[] PerlinNoiseAdvanced(int width, int height, ulong seed = 42,
            float scale = 10.0f, uint octaves = 4, float persistence = 0.5f, float lacunarity = 2.0f)
        {
            AliceError error = NativeMethods.alice_perlin_advanced(
                (UIntPtr)width, (UIntPtr)height, seed, scale, octaves, persistence, lacunarity,
                out AliceFloatBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                float[] result = new float[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_float_buffer(ref buffer);
            }
        }

        #endregion

        #region Signal Generation

        /// <summary>
        /// Generate a sine wave
        /// </summary>
        public static float[] SineWave(int n, float frequency, float amplitude = 1.0f,
            float phase = 0.0f, float dcOffset = 0.0f)
        {
            AliceError error = NativeMethods.alice_sine_wave(
                (UIntPtr)n, frequency, amplitude, phase, dcOffset, out AliceFloatBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                float[] result = new float[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_float_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Generate signal from Fourier coefficients
        /// </summary>
        public static float[] FourierGenerate(int n, FourierCoefficient[] coefficients,
            float dcOffset = 0.0f)
        {
            AliceError error = NativeMethods.alice_fourier_generate(
                (UIntPtr)n, coefficients, (UIntPtr)coefficients.Length, dcOffset,
                out AliceFloatBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                float[] result = new float[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_float_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Generate polynomial data: y = c0 + c1*x + c2*x^2 + ...
        /// </summary>
        public static float[] PolynomialGenerate(int n, double[] coefficients)
        {
            AliceError error = NativeMethods.alice_polynomial_generate(
                (UIntPtr)n, coefficients, (UIntPtr)coefficients.Length, out AliceFloatBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                float[] result = new float[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_float_buffer(ref buffer);
            }
        }

        #endregion

        #region Compression

        /// <summary>
        /// Compress data using LZMA
        /// </summary>
        /// <param name="data">Data to compress</param>
        /// <param name="preset">Compression preset (0-9)</param>
        public static byte[] LzmaCompress(byte[] data, uint preset = 6)
        {
            AliceError error = NativeMethods.alice_lzma_compress(
                data, (UIntPtr)data.Length, preset, out AliceBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                byte[] result = new byte[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Decompress LZMA data
        /// </summary>
        public static byte[] LzmaDecompress(byte[] data)
        {
            AliceError error = NativeMethods.alice_lzma_decompress(
                data, (UIntPtr)data.Length, out AliceBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                byte[] result = new byte[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Compress data using zlib
        /// </summary>
        public static byte[] ZlibCompress(byte[] data, uint level = 6)
        {
            AliceError error = NativeMethods.alice_zlib_compress(
                data, (UIntPtr)data.Length, level, out AliceBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                byte[] result = new byte[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Decompress zlib data
        /// </summary>
        public static byte[] ZlibDecompress(byte[] data)
        {
            AliceError error = NativeMethods.alice_zlib_decompress(
                data, (UIntPtr)data.Length, out AliceBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                byte[] result = new byte[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Compress float residuals with quantization
        /// </summary>
        /// <param name="residual">Float data to compress</param>
        /// <param name="bits">Quantization bits (8, 12, or 16)</param>
        /// <param name="lzmaPreset">LZMA compression preset</param>
        public static byte[] ResidualCompress(float[] residual, byte bits = 8, uint lzmaPreset = 6)
        {
            AliceError error = NativeMethods.alice_residual_compress(
                residual, (UIntPtr)residual.Length, bits, lzmaPreset, out AliceBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                byte[] result = new byte[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Decompress quantized residuals
        /// </summary>
        public static float[] ResidualDecompress(byte[] data)
        {
            AliceError error = NativeMethods.alice_residual_decompress(
                data, (UIntPtr)data.Length, out AliceFloatBuffer buffer);
            CheckError(error);

            try
            {
                int len = (int)buffer.len;
                float[] result = new float[len];
                Marshal.Copy(buffer.data, result, 0, len);
                return result;
            }
            finally
            {
                NativeMethods.alice_free_float_buffer(ref buffer);
            }
        }

        #endregion
    }
}
