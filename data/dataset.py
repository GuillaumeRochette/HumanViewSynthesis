from torch.utils.data import Dataset


class PoseDataset(Dataset):
    def __len__(self):
        """
        Returns the number of elements in the dataset.

        :return: The number of elements.
        """
        return len(self.keys)

    def __getitem__(self, item) -> dict:
        """
        Retrieve a datapoint from the dataset.

        :param item: The index of the datapoint in the dataset.
        :return: A datapoint.
        """
        key = self.keys[item]
        index = self.indexes[key]
        input = {
            "path": f"{self.path}",
            "key": f"{key}",
            "view": f"{self.view}",
            "pose_2d": {
                "p": self.poses_2d["p"][index],
                "c": self.poses_2d["c"][index],
            },
            "pose_3d": {
                "root": {
                    "p": self.poses_3d["root"]["p"][index],
                    "c": self.poses_3d["root"]["c"][index],
                },
                "relative": {
                    "p": self.poses_3d["relative"]["p"][index],
                    "c": self.poses_3d["relative"]["c"][index],
                },
            },
            "R": self.camera["R"],
            "t": self.camera["t"],
            "K": self.camera["K"],
            "dist_coef": self.camera["dist_coef"],
            "resolution": self.camera["resolution"],
        }

        return input


class ImageDataset(Dataset):
    def __len__(self):
        """
        Returns the number of elements in the dataset.

        :return: The number of elements.
        """
        return len(self.keys)

    def __getitem__(self, item) -> dict:
        """
        Retrieve a datapoint from the dataset.

        :param item: The index of the datapoint in the dataset.
        :return: A datapoint.
        """
        key = self.keys[item]
        input = {
            "path": f"{self.path}",
            "key": f"{key}",
            "view": f"{self.view}",
            "image": self.images[key],
            "mask": self.masks[key],
            "pose_2d": self.poses_2d[key],
            "pose_3d": self.poses_3d[key],
            "R": self.camera["R"],
            "t": self.camera["t"],
            "K": self.camera["K"],
            "dist_coef": self.camera["dist_coef"],
            "resolution": self.camera["resolution"],
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


class ImagePairDataset(Dataset):
    def __len__(self):
        """
        Returns the number of elements in the dataset.

        :return: The number of elements.
        """
        return len(self.keys) * len(self.pairs)

    def __getitem__(self, item) -> dict:
        """
        Retrieve a datapoint from the dataset.

        :param item: The index of the datapoint in the dataset.
        :return: A datapoint.
        """
        i, j = item // len(self.pairs), item % len(self.pairs)
        key, (view_A, view_B) = self.keys[i], self.pairs[j]
        input = {
            "path": f"{self.path}",
            "key": f"{key}",
            "A": {
                "view": f"{view_A}",
                "image": self.images[view_A][key],
                "mask": self.masks[view_A][key],
                "pose_2d": self.poses_2d[view_A][key],
                "R": self.cameras[view_A]["R"],
                "t": self.cameras[view_A]["t"],
                "K": self.cameras[view_A]["K"],
                "dist_coef": self.cameras[view_A]["dist_coef"],
                "resolution": self.cameras[view_A]["resolution"],
            },
            "B": {
                "view": f"{view_B}",
                "image": self.images[view_B][key],
                "mask": self.masks[view_B][key],
                "pose_2d": self.poses_2d[view_B][key],
                "R": self.cameras[view_B]["R"],
                "t": self.cameras[view_B]["t"],
                "K": self.cameras[view_B]["K"],
                "dist_coef": self.cameras[view_B]["dist_coef"],
                "resolution": self.cameras[view_B]["resolution"],
            },
            "W": {
                "pose_3d": self.poses_3d[key],
            },
        }

        if self.transform is not None:
            input = self.transform(input)

        return input
