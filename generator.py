import numpy
import os
from torch.utils.data import Dataset, DataLoader
import cv2

numpy.random.seed(5)


class CELEBA(Dataset):
    """
    loader for the CELEB-A dataset
    """

    def __init__(self, data_folder):
        #len is the number of files
        self.len = len(os.listdir(data_folder))
        #list of file names
        self.data_names = [os.path.join(data_folder, name) for name in sorted(os.listdir(data_folder))]

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __getitem__(self, item):
        """

        :param item: image index between 0-(len-1)
        :return: image
        """
        #load image,crop 128x128,resize,transpose(to channel first),scale (so we can use tanh)

        data = cv2.cvtColor(cv2.imread(self.data_names[item]), cv2.COLOR_BGR2RGB)
        c_x = data.shape[1] // 2
        c_y = data.shape[0] // 2
        data = data[c_y - 64:c_y + 64, c_x - 64:c_x + 64]
        data = cv2.resize(data, (64, 64))
        # CHANNEL FIRST
        data = data.transpose(2, 0, 1)
        # TANH
        data = data.astype("float32") / 255.0 * 2 - 1
        return data


if __name__ == "__main__":
    dataset = CELEBA("/home/lapis/Desktop/img_align_celeba/")
    gen = DataLoader(dataset, batch_size=3, shuffle=True)
    from matplotlib import pyplot

    for i in range(10):
        a = gen.__iter__().next()
        #scale between (0,1)
        a = (a + 1) / 2
        for el in a:
            pyplot.imshow(numpy.transpose(el.numpy(), (1, 2, 0)))
            pyplot.show()
