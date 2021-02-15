class ImageRecord:
    def __init__(self, dataset: str, file_name: str, illuminant: list, mcc_coord, img=None, extras=None):
        self.dataset = dataset
        self.file_name = file_name
        self.illuminant = illuminant
        self.mcc_coord = mcc_coord

        # BRG images
        self.img = img
        self.extras = extras

    def __repr__(self):
        r, g, b = self.illuminant[0], self.illuminant[1], self.illuminant[2]
        return "[{}, {}, ({:f}, {:f}, {:f})]".format(self.dataset, self.file_name, r, g, b)
