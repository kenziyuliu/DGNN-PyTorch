# DGNN-PyTorch

An unofficial PyTorch implementation of the paper "Skeleton-Based Action Recognition with Directed Graph Neural Networks" in CVPR 2019.

- Paper: [PDF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Skeleton-Based_Action_Recognition_With_Directed_Graph_Neural_Networks_CVPR_2019_paper.pdf)
- Code is based on 2s-AGCN: [GitHub](https://github.com/lshiwjx/2s-AGCN)

## Dependencies

- Python >= 3.5
- scipy >= 1.3.0
- numpy >= 1.16.4
- PyTorch >= 1.1.0
- tensorboardX >= 1.8   (For logging)

## Training

To start training, use the following command:

```bash
python3 main.py --config ./config/nturgbd-cross-subject/train_spatial.yaml
```

Here, `nturgbd-cross-subject` should be changed to whichever dataset/task on which to train the model e.g. `nturgbd-cross-view` or `kinetics-skeleton`.

**Note:** At the moment, only `nturgbd-cross-subject` is supported. More config files will be added later, or you could write your own config file.

## Testing

To test some model weights (by default saved in `./runs/`), do:

```bash
python3 main.py --config ./config/nturgbd-cross-subject/test_spatial.yaml
```

Similarly, change the config file as needed.

## Results

Results are being reproduced...

## TODO

- Implement data handling for 2nd (temporal) stream
- Implement support for other datasets (`nturgbd-cross-view` or `kinetics-skeleton`)
  - Modify existing code to read other datasets
  - Add more config files
