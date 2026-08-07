import re

with open("/home/koi/Documents/git/geocutool/conquer3d/csrc/ops/mc.cu", "r") as f:
    content = f.read()

# Add includes
if "<ATen/cuda/ThrustAllocator.h>" not in content:
    content = content.replace("#include <thrust/copy.h>", "#include <thrust/copy.h>\n#include <ATen/cuda/ThrustAllocator.h>\n#include <thrust/execution_policy.h>")

# Fix compute_number_active_voxels
content = re.sub(
    r"void compute_number_active_voxels.*?\{.*?uint32_t \*__restrict__ temp_buffer;\s*CHECK_CUDA_INTERNAL\(cudaMalloc\(\(void \*\*\)&temp_buffer, num_voxels \* sizeof\(uint32_t\)\)\);\s*thrust::device_ptr<uint32_t> d_prefix_sum\(temp_buffer\);\s*thrust::exclusive_scan\(active_flag_iter, active_flag_iter \+ num_voxels, d_prefix_sum\);\s*uint8_t last_flag;",
    r"""void compute_number_active_voxels(
        const uint32_t num_voxels,
        uint8_t *voxel_codes,
        uint32_t &num_active_voxels)
    {
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par.allocator(allocator);

        thrust::device_ptr<uint8_t> d_codes(voxel_codes);
        auto active_flag_iter = thrust::make_transform_iterator(d_codes, is_active_voxel());

        auto temp_buffer_t = torch::empty({num_voxels}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        uint32_t *__restrict__ temp_buffer = (uint32_t*)temp_buffer_t.data_ptr<int32_t>();
        thrust::device_ptr<uint32_t> d_prefix_sum(temp_buffer);

        thrust::exclusive_scan(policy, active_flag_iter, active_flag_iter + num_voxels, d_prefix_sum);

        uint8_t last_flag;""",
    content,
    flags=re.DOTALL
)

# Fix CHECK_CUDA_INTERNAL(cudaFree(temp_buffer));
content = content.replace("CHECK_CUDA_INTERNAL(cudaFree(temp_buffer));\n", "")

# Fix compact_active_voxels
content = re.sub(
    r"void compact_active_voxels.*?\{.*?auto counting_iter = thrust::make_counting_iterator<uint32_t>\(0\);\s*auto zip_in = thrust::make_zip_iterator\(thrust::make_tuple\(counting_iter, d_codes\)\);\s*thrust::device_ptr<uint32_t> d_out_idx\(used_voxel_index\);\s*thrust::device_ptr<uint8_t> d_out_code\(used_voxel_code\);\s*auto zip_out = thrust::make_zip_iterator\(thrust::make_tuple\(d_out_idx, d_out_code\)\);\s*thrust::copy_if\(\s*zip_in,\s*zip_in \+ num_voxels,\s*d_codes, // stencil\s*zip_out,\s*is_active_voxel\(\)\);",
    r"""void compact_active_voxels(
        const uint32_t num_voxels,
        const uint8_t *voxel_codes,
        uint32_t *used_voxel_index,
        uint8_t *used_voxel_code)
    {
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par.allocator(allocator);

        thrust::device_ptr<const uint8_t> d_codes(voxel_codes);
        auto counting_iter = thrust::make_counting_iterator<uint32_t>(0);

        auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(counting_iter, d_codes));

        thrust::device_ptr<uint32_t> d_out_idx(used_voxel_index);
        thrust::device_ptr<uint8_t> d_out_code(used_voxel_code);
        auto zip_out = thrust::make_zip_iterator(thrust::make_tuple(d_out_idx, d_out_code));

        thrust::copy_if(
            policy,
            zip_in,
            zip_in + num_voxels,
            d_codes, // stencil
            zip_out,
            is_active_voxel());""",
    content,
    flags=re.DOTALL
)

# Fix compute_unique_active_edges
content = re.sub(
    r"Edge\* compute_unique_active_edges.*?\{.*?thrust::device_ptr<Edge> d_active_edges\(active_edges\);\s*thrust::sort\(d_active_edges, d_active_edges \+ \(num_active_voxels \* 12\)\);\s*// Since 0xFFFFFFFF.*?Edge empty_edge = Edge\(0xFFFFFFFF, 0xFFFFFFFF\);\s*// Find the first.*?auto valid_end = thrust::lower_bound\(d_active_edges, d_active_edges \+ \(num_active_voxels \* 12\), empty_edge\);\s*// Deduplicate.*?auto unique_end = thrust::unique\(d_active_edges, valid_end\);\s*num_unique_edges = thrust::distance\(d_active_edges, unique_end\);\s*Edge \*__restrict__ unique_edges;\s*CHECK_CUDA_INTERNAL\(cudaMalloc\(\(void \*\*\)&unique_edges, num_unique_edges \* sizeof\(Edge\)\)\);\s*thrust::device_ptr<Edge> d_unique_edges\(unique_edges\);\s*thrust::copy\(d_active_edges, unique_end, d_unique_edges\);\s*return unique_edges;\s*}",
    r"""torch::Tensor compute_unique_active_edges(
        const uint32_t num_active_voxels,
        Edge *active_edges,
        uint32_t &num_unique_edges)
    {
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par.allocator(allocator);

        thrust::device_ptr<Edge> d_active_edges(active_edges);
        thrust::sort(policy, d_active_edges, d_active_edges + (num_active_voxels * 12));

        // Since 0xFFFFFFFF is the maximum value, the dummy edges are sorted to the END of the array.
        Edge empty_edge = Edge(0xFFFFFFFF, 0xFFFFFFFF);

        // Find the first dummy edge. Everything before this is valid!
        auto valid_end = thrust::lower_bound(policy, d_active_edges, d_active_edges + (num_active_voxels * 12), empty_edge);

        // Deduplicate the valid edges in place
        auto unique_end = thrust::unique(policy, d_active_edges, valid_end);

        num_unique_edges = thrust::distance(d_active_edges, unique_end);

        auto unique_edges_t = torch::empty({num_unique_edges * (int64_t)sizeof(Edge)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        Edge *__restrict__ unique_edges = (Edge*)unique_edges_t.data_ptr<uint8_t>();

        thrust::device_ptr<Edge> d_unique_edges(unique_edges);
        thrust::copy(policy, d_active_edges, unique_end, d_unique_edges);
        return unique_edges_t;
    }""",
    content,
    flags=re.DOTALL
)

# Fix compute_number_triangles
content = re.sub(
    r"void compute_number_triangles.*?\{.*?auto num_tris_iter = thrust::make_transform_iterator\(d_codes, num_triangles_functor\(\)\);\s*thrust::device_ptr<uint32_t> d_prefix_sum\(voxel_triangle_prefix_sums\);\s*num_triangles = thrust::reduce\(num_tris_iter, num_tris_iter \+ num_active_voxels\);\s*thrust::exclusive_scan\(num_tris_iter, num_tris_iter \+ num_active_voxels, d_prefix_sum\);\s*}",
    r"""void compute_number_triangles(
        const uint32_t num_active_voxels,
        const uint8_t *used_voxel_codes,
        uint32_t &num_triangles,
        uint32_t *voxel_triangle_prefix_sums)
    {
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par.allocator(allocator);

        thrust::device_ptr<const uint8_t> d_codes(used_voxel_codes);
        auto num_tris_iter = thrust::make_transform_iterator(d_codes, num_triangles_functor());

        thrust::device_ptr<uint32_t> d_prefix_sum(voxel_triangle_prefix_sums);
        
        num_triangles = thrust::reduce(policy, num_tris_iter, num_tris_iter + num_active_voxels);
        thrust::exclusive_scan(policy, num_tris_iter, num_tris_iter + num_active_voxels, d_prefix_sum);
    }""",
    content,
    flags=re.DOTALL
)

# Fix marching_cubes allocations
content = re.sub(
    r"uint8_t \*__restrict__ voxel_codes;\s*CHECK_CUDA_INTERNAL\(cudaMalloc\(\(void \*\*\)&voxel_codes, num_voxels \* sizeof\(uint8_t\)\)\);",
    r"auto voxel_codes_t = torch::empty({num_voxels}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));\n        uint8_t *__restrict__ voxel_codes = voxel_codes_t.data_ptr<uint8_t>();",
    content
)

content = content.replace(
    "CHECK_CUDA_INTERNAL(cudaFree(voxel_codes));\n            std::optional<torch::Tensor> out_n",
    "std::optional<torch::Tensor> out_n"
)

content = re.sub(
    r"uint32_t \*__restrict__ used_voxel_index;\s*uint8_t \*__restrict__ used_voxel_codes;\s*CHECK_CUDA_INTERNAL\(cudaMalloc\(\(void \*\*\)&used_voxel_index, num_active_voxels \* sizeof\(uint32_t\)\)\);\s*CHECK_CUDA_INTERNAL\(cudaMalloc\(\(void \*\*\)&used_voxel_codes, num_active_voxels \* sizeof\(uint8_t\)\)\);",
    r"""auto used_voxel_index_t = torch::empty({num_active_voxels}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        uint32_t *__restrict__ used_voxel_index = (uint32_t*)used_voxel_index_t.data_ptr<int32_t>();
        
        auto used_voxel_codes_t = torch::empty({num_active_voxels}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        uint8_t *__restrict__ used_voxel_codes = used_voxel_codes_t.data_ptr<uint8_t>();""",
    content
)

content = re.sub(
    r"Edge \*__restrict__ active_edges;\s*CHECK_CUDA_INTERNAL\(cudaMalloc\(\(void \*\*\)&active_edges, num_active_voxels \* 12 \* sizeof\(Edge\)\)\);",
    r"""auto active_edges_t = torch::empty({num_active_voxels * 12 * (int64_t)sizeof(Edge)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        Edge *__restrict__ active_edges = (Edge*)active_edges_t.data_ptr<uint8_t>();""",
    content
)

content = re.sub(
    r"uint32_t out_num_vertices;\s*Edge \*__restrict__ unique_edges = compute_unique_active_edges\(num_active_voxels, active_edges, out_num_vertices\);",
    r"""uint32_t out_num_vertices;
        auto unique_edges_t = compute_unique_active_edges(num_active_voxels, active_edges, out_num_vertices);
        Edge *__restrict__ unique_edges = (Edge*)unique_edges_t.data_ptr<uint8_t>();""",
    content
)

content = re.sub(
    r"uint32_t \*__restrict__ voxel_edge_to_vert_idx;\s*CHECK_CUDA_INTERNAL\(cudaMalloc\(\(void \*\*\)&voxel_edge_to_vert_idx, num_active_voxels \* 12 \* sizeof\(uint32_t\)\)\);",
    r"""auto voxel_edge_to_vert_idx_t = torch::empty({num_active_voxels * 12}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        uint32_t *__restrict__ voxel_edge_to_vert_idx = (uint32_t*)voxel_edge_to_vert_idx_t.data_ptr<int32_t>();""",
    content
)

content = re.sub(
    r"uint32_t \*__restrict__ voxel_triangle_prefix_sums;\s*CHECK_CUDA_INTERNAL\(cudaMalloc\(\(void \*\*\)&voxel_triangle_prefix_sums, num_active_voxels \* sizeof\(uint32_t\)\)\);",
    r"""auto voxel_triangle_prefix_sums_t = torch::empty({num_active_voxels}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        uint32_t *__restrict__ voxel_triangle_prefix_sums = (uint32_t*)voxel_triangle_prefix_sums_t.data_ptr<int32_t>();""",
    content
)

content = re.sub(
    r"// Cleanup\s*CHECK_CUDA_INTERNAL\(cudaFree\(voxel_codes\)\);\s*CHECK_CUDA_INTERNAL\(cudaFree\(used_voxel_index\)\);\s*CHECK_CUDA_INTERNAL\(cudaFree\(used_voxel_codes\)\);\s*CHECK_CUDA_INTERNAL\(cudaFree\(active_edges\)\);\s*CHECK_CUDA_INTERNAL\(cudaFree\(unique_edges\)\);\s*CHECK_CUDA_INTERNAL\(cudaFree\(voxel_edge_to_vert_idx\)\);\s*CHECK_CUDA_INTERNAL\(cudaFree\(voxel_triangle_prefix_sums\)\);\s*return",
    r"return",
    content,
    flags=re.DOTALL
)

with open("/home/koi/Documents/git/geocutool/conquer3d/csrc/ops/mc.cu", "w") as f:
    f.write(content)

print("mc.cu successfully modified.")
