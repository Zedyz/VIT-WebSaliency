import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import csv
from tqdm import tqdm

from config import ALPHA_FACE, ALPHA_TEXT, ALPHA_BANNER, BATCH_SIZE, WEIGHT_DECAY, DEVICE, LR, EPOCHS, IN_H, IN_W, GT_H, \
    GT_W, LAMBDA_KL, LAMBDA_TV, LAMBDA_NSS, LAMBDA_CC
from dataloader import Dataloader
from metrics import cc_weighted, total_variation_loss, cc_2, nss_2, kl_2, cc, nss, kldiv, nss_weighted, \
    kl_weighted
from model import VITModel

seed = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def build_region_map(face_720, text_720, bann_720):
    region_map = torch.ones_like(face_720)
    region_map += ALPHA_FACE * face_720
    region_map += ALPHA_TEXT * text_720
    region_map += ALPHA_BANNER * bann_720
    return region_map


def main():
    train_json = "../dataset/scanpaths_train.json"
    val_json = "../dataset/scanpaths_test.json"

    with open(train_json, 'r') as f:
        data_train = json.load(f)
    train_files = sorted({d["name"] for d in data_train if "name" in d})

    with open(val_json, 'r') as f:
        data_val = json.load(f)
    val_files = sorted({d["name"] for d in data_val if "name" in d})

    print(f"Train= {len(train_files)}, Val= {len(val_files)}")

    root = "."
    train_ds = Dataloader(root, train_files)
    val_ds = Dataloader(root, val_files)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = VITModel().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_metric = -9999.
    out_csv = "train_large_tv.csv"
    csv_file = open(out_csv, "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "epoch",
        "train_cost", "train_cc", "train_nss", "train_kl",
        "val_cost", "val_cc", "val_nss", "val_kl",
        "metric=(cc+nss-kl)"
    ])

    for epoch in range(EPOCHS):
        model.train()
        train_sum = 0.
        train_count = 0

        train_cc_sum = 0.
        train_nss_sum = 0.
        train_kl_sum = 0.
        train_count_cc = 0

        loop_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for batch in loop_train:
            img_6ch = batch["image_6ch"].to(DEVICE)
            face_720 = batch["face_full"].to(DEVICE)
            text_720 = batch["text_full"].to(DEVICE)
            bann_720 = batch["banner_full"].to(DEVICE)
            fdm_full = batch["fdm_full"].to(DEVICE)
            eye_full = batch["eye_full"].to(DEVICE)


            inp_288x512 = F.interpolate(img_6ch, (IN_H, IN_W), mode='bilinear', align_corners=False)

            pred_36x64, comps = model(inp_288x512)

            pred_up = F.interpolate(pred_36x64, (GT_H, GT_W), mode='bilinear', align_corners=False)

            region_map = build_region_map(face_720, text_720, bann_720)
            cost_cc = cc_weighted(pred_up, fdm_full, region_map)
            n_val = nss_weighted(pred_up, eye_full, region_map)
            cost_nss = (1.0 - n_val)
            kl_val = kl_weighted(pred_up, fdm_full, region_map)

            cost_tv = total_variation_loss(pred_36x64)

            cost = (LAMBDA_CC * cost_cc) + (LAMBDA_NSS * cost_nss) + (LAMBDA_KL * kl_val) + (LAMBDA_TV * cost_tv)
            optimizer.zero_grad()
            cost.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            B = img_6ch.size(0)
            train_sum += cost.item() * B
            train_count += B

            cc_val = (1.0 - cost_cc.item())
            train_cc_sum += cc_val * B
            train_nss_sum += n_val * B
            train_kl_sum += kl_val * B
            train_count_cc += B

        train_cost = train_sum / train_count
        train_cc = train_cc_sum / train_count_cc
        train_nss = train_nss_sum / train_count_cc
        train_kl = train_kl_sum / train_count_cc

        model.eval()
        val_sum = 0.
        val_count = 0

        cc_sum = 0.
        nss_sum = 0.
        kl_sum = 0.
        sample_count = 0

        loop_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for batch in loop_val:
                img_6ch = batch["image_6ch"].to(DEVICE)
                fdm_full = batch["fdm_full"].to(DEVICE)
                eye_full = batch["eye_full"].to(DEVICE)

                inp_288x512 = F.interpolate(img_6ch, (IN_H, IN_W), mode='bilinear', align_corners=False)
                pred_36x64, comps = model(inp_288x512)
                pred_up = F.interpolate(pred_36x64, (GT_H, GT_W), mode='bilinear', align_corners=False)

                c_val = cc_2(pred_up, fdm_full)
                n_val = nss_2(pred_up, eye_full)
                kl_v = kl_2(pred_up, fdm_full)
                cost_val = c_val + (LAMBDA_NSS * (1.0 - n_val)) + (LAMBDA_KL * kl_v)

                B2 = img_6ch.size(0)
                val_sum += cost_val.item() * B2
                val_count += B2

                pred_np = pred_up.squeeze(1).cpu().numpy()
                fdm_np = fdm_full.squeeze(1).cpu().numpy()
                eye_np = eye_full.squeeze(1).cpu().numpy()

                for b2 in range(B2):
                    sm = pred_np[b2]
                    mx = sm.max()
                    if mx > 1e-8:
                        sm /= mx
                    gm = fdm_np[b2]
                    fix_ = eye_np[b2]

                    c_ = cc(sm, gm)
                    nss_ = nss(sm, fix_)
                    kl_ = kldiv(sm, gm)

                    cc_sum += c_
                    nss_sum += nss_
                    kl_sum += kl_
                    sample_count += 1

        val_cost = val_sum / val_count
        val_cc = cc_sum / sample_count
        val_nss = nss_sum / sample_count
        val_kl = kl_sum / sample_count

        metric = (val_cc + val_nss) - val_kl

        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"train_cost={train_cost:.4f}, train_cc={train_cc:.4f}, train_nss={train_nss:.4f}, train_kl={train_kl:.4f} || "
              f"val_cost={val_cost:.4f}, cc={val_cc:.4f}, nss={val_nss:.4f}, kl={val_kl:.4f}, "
              f"metric={metric:.4f}")

        writer.writerow([
            epoch,
            train_cost, train_cc, train_nss, train_kl,
            val_cost, val_cc, val_nss, val_kl,
            metric
        ])
        csv_file.flush()

        # track best
        if metric > best_metric:
            best_metric = metric
            ckp = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
                "val_cc": val_cc,
                "val_nss": val_nss,
                "val_kl": val_kl
            }
            torch.save(ckp, "FERDIG_all.pth")
            print(f"saved best metric={best_metric:.4f}")

    csv_file.close()
    final_ckp = {
        "epoch": EPOCHS,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric
    }
    torch.save(final_ckp, "final_done.pth")
    print("final_done.pth")


if __name__ == "__main__":
    main()
