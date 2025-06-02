import os, argparse, cv2, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt

from config import GT_H, GT_W, IN_H, IN_W, DEVICE
from model import VITModel


def overlay_heatmap(bgr, sal, alpha=0.4):
    norm = (sal - sal.min()) / (sal.ptp() + 1e-8)
    heat = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)[..., ::-1]


def load_mask(path):
    if path and os.path.isfile(path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (GT_H, GT_W):
                m = cv2.resize(m, (GT_W, GT_H), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(m.astype(np.float32)[None] / 255.)
    return torch.zeros(1, GT_H, GT_W)


@torch.inference_mode()
def run_inference(model, img_path, face_path, text_path, banner_path,
                  show=True, save_path=None):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    if bgr.shape[:2] != (GT_H, GT_W):
        bgr = cv2.resize(bgr, (GT_W, GT_H), interpolation=cv2.INTER_AREA)

    rgb_t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                             .astype(np.float32)
                             .transpose(2, 0, 1) / 255.)
    face_t = load_mask(face_path)
    text_t = load_mask(text_path)
    banner_t = load_mask(banner_path)

    img_6ch = torch.cat([rgb_t, face_t, text_t, banner_t], 0).unsqueeze(0).to(DEVICE)

    inp = F.interpolate(img_6ch, (IN_H, IN_W), mode='bilinear', align_corners=False)
    sal36, _ = model(inp)
    sal_up = F.interpolate(sal36, (GT_H, GT_W), mode='bilinear', align_corners=False)

    overlay = overlay_heatmap(bgr, sal_up.squeeze().cpu().numpy())

    if show:
        plt.figure(figsize=(6, 10))
        plt.axis("off")
        plt.imshow(overlay)
        plt.tight_layout()
        plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("inference_results_test_set: ", save_path)


def parse_args():
    p = argparse.ArgumentParser(description="visual saliency inference_results_test_set")
    p.add_argument("--ckpt", required=True, help="Path to FERDIG_all_FINAL.pth")

    # single image
    p.add_argument("--img", help="Run on a single image")
    p.add_argument("--face_mask")
    p.add_argument("--text_mask")
    p.add_argument("--banner_mask")

    # batch mode
    p.add_argument("--img_dir")
    p.add_argument("--face_dir")
    p.add_argument("--text_dir")
    p.add_argument("--banner_dir")

    p.add_argument("--out_dir", default="inference_results_test_set")
    p.add_argument("--noshow", action="store_true")
    return p.parse_args()


def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location=DEVICE)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)


def main():
    args = parse_args()
    model = VITModel().to(DEVICE).eval()
    load_checkpoint(args.ckpt, model)

    if args.img:
        base = os.path.basename(args.img)
        face_p = args.face_mask or (os.path.join(args.face_dir, base) if args.face_dir else None)
        text_p = args.text_mask or (os.path.join(args.text_dir, base) if args.text_dir else None)
        banner_p = args.banner_mask or (os.path.join(args.banner_dir, base) if args.banner_dir else None)

        out_path = os.path.join(args.out_dir, base)
        run_inference(model, args.img, face_p, text_p, banner_p,
                      show=not args.noshow, save_path=out_path)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    imgs = [f for f in os.listdir(args.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for fn in imgs:
        run_inference(model,
                      os.path.join(args.img_dir, fn),
                      os.path.join(args.face_dir, fn) if args.face_dir else None,
                      os.path.join(args.text_dir, fn) if args.text_dir else None,
                      os.path.join(args.banner_dir, fn) if args.banner_dir else None,
                      show=False if args.noshow else True,
                      save_path=os.path.join(args.out_dir, fn))


if __name__ == "__main__":
    main()
