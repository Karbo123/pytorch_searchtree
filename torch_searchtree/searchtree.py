import torch
import taichi as ti

ti.init(arch=ti.cpu)


@ti.data_oriented
class SearchTree:
    def __init__(self, points, voxel_size=1e-4, kwargs_partition=dict(method="octree", depth=8, ratio=3)):
        assert isinstance(points, torch.Tensor), "`points` should be a torch.Tensor"
        assert points.ndim == 2 and points.size(1) == 3, "`points` has a wrong size"
        num_points = len(points)
        #
        bound_min = points.min(dim=0).values
        bound_max = points.max(dim=0).values
        points_voxelized = (points - bound_min).div(voxel_size).floor().int()
        size_xyz = (bound_max - bound_min).div(voxel_size).ceil().long()

        # check
        cnt = points_voxelized.unique(dim=0, return_counts=True)[1]
        assert cnt.max() == 1, "more than one point within one cell" # NOTE TODO

        #
        self.size_xyz = tuple(size_xyz.tolist())
        self.kwargs_partition = kwargs_partition

        #
        self.partition()

        # NOTE the leaf node only has one point index
        self.point_index = ti.field(dtype=ti.i32)  # point index contained in a tree node
        self.blocks[-1].place(self.point_index)

        #
        self.insert(points_voxelized)  # insert to sparse tree



    def partition(self, ):

        method = self.kwargs_partition["method"]
        assert method in ("octree", ), f"unknown partition: {method}"

        if method == "octree":
            depth = self.kwargs_partition["depth"]
            ratio = self.kwargs_partition["ratio"]

            self.blocks = [None for _ in range(depth)]
            for ind in range(depth):
                node = ti.root if ind == 0 else self.blocks[ind - 1]
                self.blocks[ind] = getattr(node, "pointer" if ind < depth - 1 else "bitmasked")(ti.ijk, ratio)

        else:
            raise NotImplementedError

    @ti.kernel
    def insert(self, voxelized_index : ti.ext_arr()):
        num = voxelized_index.shape[0]
        for ind in range(num):  # struct-fors
            i = voxelized_index[ind, 0]
            j = voxelized_index[ind, 1]
            k = voxelized_index[ind, 2]
            self.point_index[i, j, k] = ind # NOTE race condition if cnt > 1 ???


    def self_nearest(self, ):
        raise NotImplementedError




if __name__ == "__main__":
    points = torch.randn(2048, 3)
    points = torch.nn.functional.normalize(points, dim=1)
    tree = SearchTree(points)
    
    import ipdb; ipdb.set_trace()
    print()
