/**
 * ALICE-Zip Unity Texture Generator
 *
 * Unity-specific helpers for generating textures from ALICE-Zip data.
 */

using UnityEngine;

namespace AliceZip.Unity
{
    /// <summary>
    /// Unity-specific texture generation utilities
    /// </summary>
    public static class AliceTextureGenerator
    {
        /// <summary>
        /// Generate a Perlin noise Texture2D
        /// </summary>
        /// <param name="width">Texture width</param>
        /// <param name="height">Texture height</param>
        /// <param name="seed">Random seed for reproducibility</param>
        /// <param name="scale">Noise scale (larger = more zoomed out)</param>
        /// <param name="octaves">Number of octaves for fractal noise</param>
        /// <param name="format">Texture format</param>
        /// <returns>Generated Texture2D</returns>
        public static Texture2D GeneratePerlinTexture(
            int width, int height,
            ulong seed = 42,
            float scale = 10.0f,
            uint octaves = 4,
            TextureFormat format = TextureFormat.R8)
        {
            float[] noise = Alice.PerlinNoise2D(width, height, seed, scale, octaves);
            return FloatArrayToTexture(noise, width, height, format);
        }

        /// <summary>
        /// Generate an advanced Perlin noise Texture2D
        /// </summary>
        public static Texture2D GeneratePerlinTextureAdvanced(
            int width, int height,
            ulong seed = 42,
            float scale = 10.0f,
            uint octaves = 4,
            float persistence = 0.5f,
            float lacunarity = 2.0f,
            TextureFormat format = TextureFormat.R8)
        {
            float[] noise = Alice.PerlinNoiseAdvanced(
                width, height, seed, scale, octaves, persistence, lacunarity);
            return FloatArrayToTexture(noise, width, height, format);
        }

        /// <summary>
        /// Convert a float array to a grayscale Texture2D
        /// </summary>
        /// <param name="data">Float array (0.0 to 1.0 range)</param>
        /// <param name="width">Texture width</param>
        /// <param name="height">Texture height</param>
        /// <param name="format">Texture format</param>
        /// <returns>Generated Texture2D</returns>
        public static Texture2D FloatArrayToTexture(
            float[] data, int width, int height,
            TextureFormat format = TextureFormat.R8)
        {
            Texture2D texture = new Texture2D(width, height, format, false);

            if (format == TextureFormat.R8)
            {
                byte[] bytes = new byte[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    bytes[i] = (byte)(Mathf.Clamp01(data[i]) * 255);
                }
                texture.LoadRawTextureData(bytes);
            }
            else if (format == TextureFormat.RFloat)
            {
                texture.SetPixelData(data, 0);
            }
            else
            {
                // Generic fallback using SetPixels
                Color[] colors = new Color[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    float v = Mathf.Clamp01(data[i]);
                    colors[i] = new Color(v, v, v, 1.0f);
                }
                texture.SetPixels(colors);
            }

            texture.Apply();
            return texture;
        }

        /// <summary>
        /// Convert a float array to a colored Texture2D using a gradient
        /// </summary>
        /// <param name="data">Float array (0.0 to 1.0 range)</param>
        /// <param name="width">Texture width</param>
        /// <param name="height">Texture height</param>
        /// <param name="gradient">Color gradient to apply</param>
        /// <returns>Generated Texture2D</returns>
        public static Texture2D FloatArrayToTexture(
            float[] data, int width, int height, Gradient gradient)
        {
            Texture2D texture = new Texture2D(width, height, TextureFormat.RGBA32, false);

            Color[] colors = new Color[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                colors[i] = gradient.Evaluate(Mathf.Clamp01(data[i]));
            }
            texture.SetPixels(colors);
            texture.Apply();

            return texture;
        }

        /// <summary>
        /// Generate a normal map from a height map
        /// </summary>
        /// <param name="heightMap">Height map float array</param>
        /// <param name="width">Texture width</param>
        /// <param name="height">Texture height</param>
        /// <param name="strength">Normal strength multiplier</param>
        /// <returns>Normal map Texture2D</returns>
        public static Texture2D HeightMapToNormalMap(
            float[] heightMap, int width, int height, float strength = 1.0f)
        {
            Texture2D normalMap = new Texture2D(width, height, TextureFormat.RGBA32, false);
            Color[] normals = new Color[width * height];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Sample neighboring pixels
                    float left = GetHeight(heightMap, x - 1, y, width, height);
                    float right = GetHeight(heightMap, x + 1, y, width, height);
                    float up = GetHeight(heightMap, x, y - 1, width, height);
                    float down = GetHeight(heightMap, x, y + 1, width, height);

                    // Calculate normal
                    float dx = (right - left) * strength;
                    float dy = (down - up) * strength;

                    Vector3 normal = new Vector3(-dx, -dy, 1.0f).normalized;

                    // Convert to color (0-1 range)
                    normals[y * width + x] = new Color(
                        normal.x * 0.5f + 0.5f,
                        normal.y * 0.5f + 0.5f,
                        normal.z * 0.5f + 0.5f,
                        1.0f
                    );
                }
            }

            normalMap.SetPixels(normals);
            normalMap.Apply();
            return normalMap;
        }

        private static float GetHeight(float[] heightMap, int x, int y, int width, int height)
        {
            x = Mathf.Clamp(x, 0, width - 1);
            y = Mathf.Clamp(y, 0, height - 1);
            return heightMap[y * width + x];
        }
    }
}
