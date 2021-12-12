import torch
from math import log, ceil
import taichi as ti

ti.init(arch=ti.cpu)


@ti.data_oriented
class SearchTree:
    def __init__(self, 
                 points, # the point cloud
                 voxel_size=1e-2, # the size to voxelized
                 num_max_voxel_point=512, # the maximum num of points within one voxel
                 kwargs_partition=dict(method="octree", depth=2, ratio=2), # how the tree is constructed
                ):
        assert isinstance(points, torch.Tensor), "`points` should be a torch.Tensor"
        assert points.ndim == 2 and points.size(0) >= 1 and points.size(1) == 3, "`points` has a wrong size"
        assert points.dtype == torch.float32, "`points` has a wrong datatype"

        self.num_points = len(points)
        self.voxel_size = voxel_size
        self.num_max_voxel_point = num_max_voxel_point
        self.kwargs_partition = kwargs_partition
        
        # bound
        bound_min = points.min(dim=0).values
        bound_max = points.max(dim=0).values
        self.bound_min = ti.Vector(bound_min.tolist())
        self.bound_max = ti.Vector(bound_max.tolist())

        # point cloud
        self.points = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_points, ))
        self.points.from_torch(points)

        self.size_xyz = tuple((bound_max - bound_min).div(voxel_size).ceil().long().tolist())

        #
        self.partition()

        # NOTE the leaf node only has one point index
        self.point_index = ti.Vector.field(1 + num_max_voxel_point, dtype=ti.i32) # point index contained in a tree node
        self.blocks[-1].place(self.point_index)
        #
        self.insert()  # insert to sparse tree


    def partition(self, ):

        method = self.kwargs_partition["method"]
        assert method in ("octree", ), f"unknown partition: {method}"

        if method == "octree":
            depth = self.kwargs_partition["depth"]
            ratio = self.kwargs_partition["ratio"]
            self.bottom_size = ratio # the size of the bottom block

            padded_size = tuple(pow(ratio, ceil(log(s, ratio))) for s in self.size_xyz)

            self.blocks = [None for _ in range(depth)]
            for ind in range(depth):
                node = ti.root if ind == 0 else self.blocks[ind - 1]
                self.blocks[ind] = getattr(node, "pointer" if ind < depth - 1 else "bitmasked") \
                                          (ti.ijk, tuple(max(1, pad // pow(ratio, depth - 1)) for pad in padded_size) \
                                                   if ind == 0 else ratio)

        else:
            raise NotImplementedError


    @ti.kernel
    def insert(self, ):
        for ind in self.points:  # struct-fors
            ijk = ti.cast(ti.floor((self.points[ind] - self.bound_min) / self.voxel_size), ti.i32)
            ind_insert = ti.atomic_add(self.point_index[ijk][0], 1)
            assert ind_insert < self.num_max_voxel_point, "num of points exceed limit for a voxel"
            # https://docs.taichi.graphics/lang/articles/advanced/meta#when-to-use-tistatic-with-for-loops
            for ind_vec in ti.static(range(1, self.num_max_voxel_point + 1)):
                if ind_vec == ind_insert + 1:
                    self.point_index[ijk][ind_vec] = ind + 1 # 0 is nothing
        

    @ti.func
    def distance(self, vec3A, vec3B):
        return sum((vec3A - vec3B) ** 2)


    @ti.func
    def cell_nearest(self, 
                     block_index, # the bottom block index (3d)
                     query_index, # the query point index (1d)
                    ):
        best_dist = 1e9
        best_index = 0
        for offset_ijk in ti.grouped(ti.ndrange(self.bottom_size, self.bottom_size, self.bottom_size)):
            vec = self.point_index[block_index * self.bottom_size + offset_ijk]
            num = vec[0]
            # https://docs.taichi.graphics/lang/articles/advanced/meta#when-to-use-tistatic-with-for-loops
            for ith in ti.static(range(1, self.num_max_voxel_point + 1)):
                if ith >= num + 1: continue
                key_index = vec[ith] - 1
                if key_index == query_index: continue
                dist = self.distance(self.points[key_index], self.points[query_index])
                if dist < best_dist:
                    best_dist = dist
                    best_index = key_index
        return best_index, best_dist


    def self_nearest(self, ):
        self.result = ti.Struct.field(dict(index=ti.i32, dist=ti.f32), shape=(self.num_points, ))
        self._self_nearest()
        return self.result.to_torch()


    @ti.kernel
    def _self_nearest(self, ):
        for query_index in self.points:
            point_index = ti.cast(ti.floor((self.points[query_index] - self.bound_min) / self.voxel_size), ti.i32)
            block_index = ti.rescale_index(self.point_index, self.blocks[-1], point_index) # NOTE TODO only self block
            best_index, best_dist = self.cell_nearest(block_index, query_index)
            self.result[query_index].index = best_index
            self.result[query_index].dist = best_dist
        
            




if __name__ == "__main__":
    points = torch.randn(2048, 3)
    points = torch.nn.functional.normalize(points, dim=1)
    tree = SearchTree(points)
    xx = tree.self_nearest()
    import ipdb; ipdb.set_trace()
    print()
