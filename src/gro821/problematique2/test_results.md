# GRO821 - Tests Results

## "Root" AABB

Use AABBs to reject rays before primitive tests. Reduces the number of intersection tests.

### Test: Sphere_0

Configuration:

```py
SHADING = "gouraud"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
scene.append(add_sphere((1, 1, 10), 1.0, (0.8, 0.8, 0.8)))
```

Results:

| Metric | Before | After |
| --- | --- | --- |
| Render | 0.844 s | 1.513 s |
| Intersection tests | 79619 (2.65 / px) | 57840 (1.93 / px) |

### Test: Bunny_res4_0

Configuration:

```py
SHADING = "flat"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
for i in random.sample(range(len(FACES)), len(FACES) // 10): # 10% sampled
    scene.append(add_triangle(i, (0.8, 0.8, 0.8)))
```

Results:

| Metric | Before | After |
| --- | --- | --- |
| Render | ~80 s | 21.237 s |
| Intersection tests | ~4.1M (~135 / px) | 1277163 (42.57 / px) |

> N.B. Results without "root" AABB were not recorded, and are an estimation.

### Conclusion - "root" AABB

The number of intersection tests is greatly reduced, so the "root" AABB optimization is retained and becomes the new baseline.

## Flat-shading VS Gouraud

The biggest performance gain with the shading comes from precomputing the triangle quantities before starting to render the scene (along with changes to the data model to use indices to shared NumPy arrays instead of per-triangle copies).

## Test: Sphere_1

Configuration:

```py
SHADING = "flat", "gouraud"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
scene.append(add_sphere((1, 1, 10), 1.0, (0.8, 0.6, 1.0)))
```

Results:

| Metric | Flat | Gouraud |
| --- | --- | --- |
| Render | 1.598 s | 1.576 s |
| Intersection tests | 57840 (1.93 / px) | 57840 (1.93 / px) |

## Test: Bunny_res4_1

Configuration:

```py
SHADING = "flat", "gouraud"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
for i in random.sample(range(len(FACES)), len(FACES) // 10): # 10% sampled
    scene.append(add_triangle(i, (0.8, 0.8, 0.8)))
```

Results:

| Metric | Flat | Gouraud |
| --- | --- | --- |
| Render | 20.681 s | 20.637 s |
| Intersection tests | 1231995 (41.07 / px) | 1228589 (40.95 / px) |

### Conclusion - Shading Model

The number of intersection tests is roughly equivalent between the two shading models, so for simplicity, flat-shading will be used for the baseline.

## Octree

Build an Octree of the scene objects to reject rays before primitive tests. Reduces the number of intersection tests, at the cost of memory and compute time to build the octree.

### Test: Sphere_2

Configuration:

```py
SHADING = "flat"
ACCELERATION = "root_aabb", "octree"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
scene.append(add_sphere((1, 1, 10), 1.0, (0.8, 0.6, 1.0)))
```

Results:

| Metric | AABB | Octree |
| --- | --- | --- |
| Render | 1.564 s | 1.684 s |
| Intersection tests | 57840 (1.93 / px) | 57840 (1.93 / px) |

### Test: Bunny_res4_2

Configuration:

```py
SHADING = "flat"
ACCELERATION = "root_aabb", "octree"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
for i in random.sample(range(len(FACES)), len(FACES) // 10): # 10% sampled
    scene.append(add_triangle(i, (0.8, 0.8, 0.8)))
```

Results:

| Metric | AABB | Octree |
| --- | --- | --- |
| Render | 21.673 s | 11.005 s |
| Intersection tests | 1256698 (41.89 / px) | 456594 (15.22 / px) |

### Test: Bunny_res4_3

Configuration:

PLY: 453 vertices, 948 faces

```py
SHADING = "flat"
ACCELERATION = "root_aabb", "octree"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
for i in random.sample(range(len(FACES)), len(FACES) // 2): # 50% sampled
    scene.append(add_triangle(i, (0.8, 0.8, 0.8)))
```

Results:

| Metric | AABB | Octree |
| --- | --- | --- |
| Render | 126.215 s | 54.231 s |
| Intersection tests | 7911941 (263.73 / px) | 2950037 (98.33 / px) |

### Test: Bunny_res3_0

Configuration:

PLY: 1889 vertices, 3851 faces

```py
SHADING = "flat"
ACCELERATION = "root_aabb", "octree"
bunny.load(f"{root}/meshes/bun_zipper_res3.ply", scale=5.0)
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
for i in random.sample(range(len(FACES)), len(FACES) // 10): # 10% sampled
    scene.append(add_triangle(i, (0.8, 0.8, 0.8)))
```

Results:

| Metric | AABB | Octree |
| --- | --- | --- |
| Render | 81.231 s | 23.102 s |
| Intersection tests | 5163246 (172.11 / px) | 1083517 (36.12 / px) |

### Conclusion - Octree

The number of intersection tests is greatly reduced, so the octree optimization is retained. However, the "root" AABB method remains the baseline for now.

## BVH

Build a BVH of the scene objects to reject rays before primitive tests. Reduces the number of intersection tests, at the cost of memory and compute time to build the BVH.

### Test: Bunny_res4_4

Configuration:

PLY: 453 vertices, 948 faces

```py
SHADING = "flat"
ACCELERATION = "root_aabb", "octree", "bvh"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
for i in random.sample(range(len(FACES)), len(FACES) // 10): # 10% sampled
    scene.append(add_triangle(i, (0.8, 0.8, 0.8)))
```

Results:

| Metric | AABB | Octree | BVH |
| --- | --- | --- | --- |
| Render | 21.157 s | 10.944 s | 5.006 s |
| Intersection tests | 1276869 (42.56 / px) | 461078 (15.37 / px) | 92532 (3.08 / px) |

### Test: Bunny_res3_1

Configuration:

PLY: 1889 vertices, 3851 faces

```py
SHADING = "flat"
bunny.load(f"{root}/meshes/bun_zipper_res3.ply", scale=5.0)
ACCELERATION = "root_aabb", "octree", "bvh"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
for i in random.sample(range(len(FACES)), len(FACES) // 10): # 10% sampled
    scene.append(add_triangle(i, (0.8, 0.8, 0.8)))
```

Results:

| Metric | AABB | Octree | BVH |
| --- | --- | --- | --- |
| Render | 84.266 s | 24.382 s | 8.446 s |
| Intersection tests | 5254959 (175.17 / px) | 1120675 (37.36 / px) | 109061 (3.64 / px) |

### Test: Bunny_res3_2

Configuration:

PLY: 1889 vertices, 3851 faces

```py
SHADING = "flat"
bunny.load(f"{root}/meshes/bun_zipper_res3.ply", scale=5.0)
ACCELERATION = "root_aabb", "octree", "bvh"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
for i in random.sample(range(len(FACES)), len(FACES) // 2): # 50% sampled
    scene.append(add_triangle(i, (0.8, 0.8, 0.8)))
```

Results:

| Metric | AABB | Octree | BVH |
| --- | --- | --- | --- |
| Render | 499.819 s | 115.487 s | 20.747 s |
| Intersection tests | 32703762 (1090.13 / px) | 6473140 (215.77 / px) | 264073 (8.80 / px) |

> N.B. `bun_zipper_res3` with 100% of samples using BVH yielded a render time of 33.378 s with 380753 tests (12.69 / px).

### Conclusion - BVH

The number of intersection tests is greatly reduced, and seem to scale better with the number of triangles, so the BVH optimization is retained, and becomes the new baseline.
