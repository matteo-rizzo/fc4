from classes.data.datasets.GehlerDataset import GehlerDataset


def main():
    ds = GehlerDataset()
    ds.regenerate_meta_data()
    ds.regenerate_image_packs()


if __name__ == '__main__':
    main()
