# GRO821 - Tests Results

## "Root" AABB

Use AABBs to reject rays before primitive tests. Reduces the number of intersection tests.

### Test: Sphere

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

### Test: Bunny

Configuration:

```py
SHADING = "flat"
scene = [add_plane((0.0, -0.5, 0.0), (0.0, 1.0, 0.0))]
for i in random.sample(range(len(FACES)), len(FACES) // 10):
    scene.append(add_triangle(i, (0.8, 0.8, 0.8)))
```

Results:

| Metric | Before | After |
| --- | --- | --- |
| Render | ~80 s | 21.237 s |
| Intersection tests | ~4.1M (~135 / px) | 1277163 (42.57 / px) |

Conclusion:

The number of intersection tests is greatly reduced, so the "root" AABB optimization is retained and becomes the new baseline.

## Octree

Build an Octree of the scene objects to reject trays before primitive tests. Reduces the number of intersection tests, at the cost of memory and compute time to build the octree.


