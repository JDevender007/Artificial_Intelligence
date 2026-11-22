from sdlora.report import write_reports
from sdlora.safety import evaluate_prompt_filter


def test_write_reports(tmp_path):
    variants = ["base", "lora_r8@10", "lora_r8@20"]
    rows, prompts = [], []
    for v_i, v in enumerate(variants):
        for s in (0, 1):
            rows.append({"variant": v, "seed": s, "n_images": 3, "fid": 100.0 - 10 * v_i + s, "kid_x1000": 5.0,
                         "kid_std_x1000": 1.0, "clip_score": 30.0 + v_i, "top1": 0.1, "top5": 0.3,
                         "mem_max_sim": 0.8, "mem_frac_ge_095": 0.0, "sd_nsfw_rate": 0.0, "clip_flag_rate": 0.0,
                         "gen_s_per_img": 1.0})
            for i in range(3):
                prompts.append({"variant": v, "seed": s, "idx": i, "clip_score": 30.0 + v_i + i, "top1": 0.0,
                                "top5": 1.0, "mem_sim": 0.8})
    ref = {"fid": 20.0, "kid_x1000": 1.0, "clip_score": 31.0, "top1": 0.2, "top5": 0.5, "mem_max_sim": 0.85,
           "mem_frac_ge_095": 0.0, "clip_flag_rate": 0.0}
    safety = {"red_team": evaluate_prompt_filter(), "captions": {"n": 6, "false_positive_rate": 0.0, "blocked": []},
              "image_filters": [{"images": "real", "n": 3, "sd_nsfw_rate": None, "clip_flag_rate": 0.0}]}
    train = [{"seed": 0, "steps": 20, "lora_params": 797184, "train_s": 12.0, "s_per_step": 0.6,
              "final_loss_avg50": 0.1}]
    res = write_reports(tmp_path, rows, prompts, ref, safety, train, variants, n_bootstrap=100)
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert md.index("| base") < md.index("| lora_r8@10") < md.index("| lora_r8@20")
    boot = (tmp_path / "bootstrap.md").read_text(encoding="utf-8")
    assert "lora_r8@20 - base | clip_score | +2.0000" in boot and "lora_r8@20 - lora_r8@10" in boot
    assert "n/a" in (tmp_path / "safety.md").read_text(encoding="utf-8")
    assert "797,184" in (tmp_path / "training.md").read_text(encoding="utf-8")
    assert len(res["bootstrap"]) == 8
