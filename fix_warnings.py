import re

with open("/home/koi/Documents/git/geocutool/conquer3d/csrc/ops/mc.cu", "r") as f:
    content = f.read()

# Fix narrowing conversions in torch::empty
content = re.sub(
    r"torch::empty\(\{num_voxels\}",
    r"torch::empty({(int64_t)num_voxels}",
    content
)
content = re.sub(
    r"torch::empty\(\{num_active_voxels\}",
    r"torch::empty({(int64_t)num_active_voxels}",
    content
)
content = re.sub(
    r"torch::empty\(\{num_active_voxels \* 12 \* \(int64_t\)sizeof\(Edge\)\}",
    r"torch::empty({(int64_t)(num_active_voxels * 12 * sizeof(Edge))}",
    content
)
content = re.sub(
    r"torch::empty\(\{num_active_voxels \* 12\}",
    r"torch::empty({(int64_t)(num_active_voxels * 12)}",
    content
)
content = re.sub(
    r"torch::empty\(\{num_unique_edges \* \(int64_t\)sizeof\(Edge\)\}",
    r"torch::empty({(int64_t)(num_unique_edges * sizeof(Edge))}",
    content
)
content = re.sub(
    r"torch::empty\(\{out_num_vertices, 3\}",
    r"torch::empty({(int64_t)out_num_vertices, 3}",
    content
)
content = re.sub(
    r"torch::empty\(\{out_num_triangles, 3\}",
    r"torch::empty({(int64_t)out_num_triangles, 3}",
    content
)
content = re.sub(
    r"torch::empty\(\{out_num_vertices, 2\}",
    r"torch::empty({(int64_t)out_num_vertices, 2}",
    content
)

# Fix std::nullopt deduction
content = content.replace(
    "std::nullopt\n            );",
    "std::optional<torch::Tensor>(std::nullopt)\n            );"
)

with open("/home/koi/Documents/git/geocutool/conquer3d/csrc/ops/mc.cu", "w") as f:
    f.write(content)
