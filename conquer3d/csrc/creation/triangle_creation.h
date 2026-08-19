/**
 * @file triangle_creation.h
 * @brief Procedural canonical primitive generators in C++ (UV Sphere, Regular Tetrahedron).
 */

#ifndef TRIANGLE_CREATION_H
#define TRIANGLE_CREATION_H

#include <torch/extension.h>
#include <tuple>
#include <vector>
#include <cmath>

namespace triangle_creation
{
    /**
     * @brief Generates a parametric UV Sphere mesh with vertex coordinates and triangle faces on CPU.
     * 
     * @param[in] sectors Number of longitudinal angular sectors (default: 32).
     * @param[in] stacks  Number of latitudinal horizontal stacks (default: 16).
     * @param[in] radius  Radius of the generated sphere (default: 1.0f).
     * 
     * @return Tuple containing (vertices [N, 3] float32, triangles [M, 3] int32).
     */
    inline std::tuple<torch::Tensor, torch::Tensor> create_sphere(int sectors = 32, int stacks = 16, float radius = 1.0f) {
        std::vector<float> vertices;
        std::vector<int> triangles;

        float sectorStep = 2 * M_PI / sectors;
        float stackStep = M_PI / stacks;

        // 1. North Pole
        vertices.push_back(0.0f);
        vertices.push_back(0.0f);
        vertices.push_back(radius);

        // 2. Rings (stacks-1)
        for (int i = 1; i < stacks; ++i) {
            float stackAngle = M_PI / 2 - i * stackStep;
            float xy = radius * cosf(stackAngle);
            float z = radius * sinf(stackAngle);

            for (int j = 0; j < sectors; ++j) {
                float sectorAngle = j * sectorStep;
                vertices.push_back(xy * cosf(sectorAngle));
                vertices.push_back(xy * sinf(sectorAngle));
                vertices.push_back(z);
            }
        }

        // 3. South Pole
        vertices.push_back(0.0f);
        vertices.push_back(0.0f);
        vertices.push_back(-radius);

        // Generate Triangles
        // North pole triangles
        for (int j = 0; j < sectors; ++j) {
            int next_j = (j + 1) % sectors;
            triangles.push_back(0); // North pole is index 0
            triangles.push_back(1 + j);
            triangles.push_back(1 + next_j);
        }

        // Middle ring triangles
        for (int i = 0; i < stacks - 2; ++i) {
            int current_ring_start = 1 + i * sectors;
            int next_ring_start = 1 + (i + 1) * sectors;

            for (int j = 0; j < sectors; ++j) {
                int next_j = (j + 1) % sectors;
                
                int v0 = current_ring_start + j;
                int v1 = current_ring_start + next_j;
                int v2 = next_ring_start + j;
                int v3 = next_ring_start + next_j;

                triangles.push_back(v0);
                triangles.push_back(v2);
                triangles.push_back(v1);

                triangles.push_back(v1);
                triangles.push_back(v2);
                triangles.push_back(v3);
            }
        }

        // South pole triangles
        int south_pole_index = 1 + (stacks - 1) * sectors;
        int last_ring_start = 1 + (stacks - 2) * sectors;
        for (int j = 0; j < sectors; ++j) {
            int next_j = (j + 1) % sectors;
            triangles.push_back(south_pole_index);
            triangles.push_back(last_ring_start + next_j);
            triangles.push_back(last_ring_start + j);
        }

        auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32);
        auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32);

        torch::Tensor V = torch::from_blob(vertices.data(), {(int)vertices.size() / 3, 3}, opts_f32).clone();
        torch::Tensor F = torch::from_blob(triangles.data(), {(int)triangles.size() / 3, 3}, opts_i32).clone();

        return {V, F};
    }

    /**
     * @brief Generates a regular canonical tetrahedron mesh with 4 vertices and 4 faces on CPU.
     * 
     * @param[in] radius Circumscribed sphere radius of the tetrahedron (default: 1.0f).
     * 
     * @return Tuple containing (vertices [4, 3] float32, triangles [4, 3] int32).
     */
    inline std::tuple<torch::Tensor, torch::Tensor> create_tetrahedra(float radius = 1.0f) {
        float a = radius / std::sqrt(3.0f);
        std::vector<float> vertices = {
             a,  a,  a,
            -a, -a,  a,
            -a,  a, -a,
             a, -a, -a
        };

        std::vector<int> triangles = {
            0, 2, 1,
            0, 1, 3,
            0, 3, 2,
            1, 2, 3
        };

        auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32);
        auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32);

        torch::Tensor V = torch::from_blob(vertices.data(), {4, 3}, opts_f32).clone();
        torch::Tensor F = torch::from_blob(triangles.data(), {4, 3}, opts_i32).clone();

        return {V, F};
    }
}

#endif // TRIANGLE_CREATION_H
