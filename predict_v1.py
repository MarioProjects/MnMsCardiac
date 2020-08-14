#!/usr/bin/env python
# coding: utf-8

# ---- Library import ----
from torch.utils.data import DataLoader

# ---- My utils ----
from models import *
from utils.arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.dataload import dataset_selector, save_nii
from utils.testing import *

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    os.makedirs(args.output_data_directory, exist_ok=True)
    if args.eval_overlays:
        os.makedirs(os.path.join(args.output_data_directory, "preds_overlays"), exist_ok=True)

    aug, aug_img, _ = data_augmentation_selector(args.data_augmentation, args.img_size, args.crop_size)

    model = model_selector(args.model_name, in_channels=1)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if args.model_checkpoint != "":
        print("Load from pretrained checkpoint: {}".format(args.model_checkpoint))
        model.load_state_dict(torch.load(args.model_checkpoint))
    else:
        assert False, "Please specify a model checkpoint!"

    test_dataset = dataset_selector(None, None, aug, args)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        drop_last=False, collate_fn=test_dataset.simple_collate
    )

    model.eval()
    with torch.no_grad():
        for (ed_volume, es_volume, img_affine, img_header, img_shape, img_id, original_ed, original_es) in test_loader:

            ed_volume = ed_volume.type(torch.float).cuda()
            es_volume = es_volume.type(torch.float).cuda()

            prob_pred_ed = model(ed_volume)
            prob_pred_es = model(es_volume)

            pred_ed = binarize_volume_prediction(prob_pred_ed, img_shape)  # [slices, height, width]
            pred_es = binarize_volume_prediction(prob_pred_es, img_shape)  # [slices, height, width]

            pred_ed = pred_ed.transpose(1, 2, 0)  # [height, width, slices]
            pred_es = pred_es.transpose(1, 2, 0)  # [height, width, slices]

            save_nii(
                os.path.join(args.output_data_directory, "{}_sa_ED.nii.gz".format(img_id)),
                pred_ed, img_affine, img_header
            )

            save_nii(
                os.path.join(args.output_data_directory, "{}_sa_ES.nii.gz".format(img_id)),
                pred_es, img_affine, img_header
            )

            if args.eval_overlays_path != "none":
                plot_save_pred_volume(original_ed, pred_ed, args.eval_overlays_path, "{}_ed".format(img_id))
                plot_save_pred_volume(original_ed, pred_ed, args.eval_overlays_path, "{}_es".format(img_id))
