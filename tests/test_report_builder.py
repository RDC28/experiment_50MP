"""Conclusions of the run report, checked on the numbers of the first full Colab run (L4, 10852 x 4604)."""

from synthetic_raw_lab import report_builder


def consistency(psnr, ssim, ms_ssim, edges, delta_e, wasserstein):
    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "ms_ssim": {"value": ms_ssim},
        "edge_displacement": {"mean_px": edges},
        "color_difference_ciede2000": {"mean": delta_e},
        "luminance_distribution": {"wasserstein_1": wasserstein},
    }


def statistics(minimum, below, above):
    return {
        "min": {"R": minimum, "G": minimum, "B": minimum},
        "below_zero_fraction": below,
        "above_one_fraction": above,
    }


def colab_manifest():
    return {
        "status": "complete",
        "config": {"input_category": "universal"},
        "baselines": {
            "lanczos": {
                "source_consistency": consistency(55.23, 0.99853, 0.99981, 0.450, 0.1380, 0.0047),
                "statistics": statistics(-0.0019, 0.004226, 0.000528),
            },
            "bicubic": {
                "source_consistency": consistency(54.93, 0.99835, 0.99980, 0.451, 0.1456, 0.0051),
                "statistics": statistics(-0.0012, 0.004111, 0.000476),
            },
        },
        "super_resolution": {
            "modes": {
                "conservative": {
                    "source_consistency": consistency(66.91, 0.99987, 0.99998, 0.099, 0.0417, 0.0023),
                    "statistics": statistics(-0.0033, 0.005068, 0.000734),
                    "source_projection": {"strength": 1.0},
                },
                "balanced": {
                    "source_consistency": consistency(53.21, 0.99793, 0.99978, 0.317, 0.1624, 0.0045),
                    "statistics": statistics(-0.1701, 0.004047, 0.000450),
                    "source_projection": {"strength": 1.0},
                },
            }
        },
    }


def test_every_output_is_ranked_on_every_metric_with_the_correction_caveat():
    lines = report_builder.conclusions(colab_manifest())
    assert (
        "Source PSNR (dB) (higher is better), best first: sr_conservative* 66.91, lanczos 55.23, bicubic 54.93, "
        "sr_balanced* 53.21." in lines
    )
    assert (
        "delta E 2000 mean (lower is better), best first: sr_conservative* 0.0417, lanczos 0.1380, bicubic 0.1456, "
        "sr_balanced* 0.1624." in lines
    )
    assert any(
        line.startswith("Edge displacement mean (px) (lower is better), best first: sr_conservative*") for line in lines
    )
    assert any("partly true by construction" in line and line.startswith("* ") for line in lines)
    assert not any("Highest source PSNR: lanczos" in line for line in lines)
    assert any("the lowest linear value is -0.1701 (sr_balanced)" in line for line in lines)


def test_the_final_image_is_stated_with_its_reason():
    manifest = colab_manifest()
    manifest["final_output"] = {
        "method": "sr_conservative",
        "rule": "auto",
        "reason": "first eligible output, because the primary face is 700x700 source px",
        "skipped": [],
        "notes": [],
        "files": {"tiff16": {"path": "/x/final/final_scene_sr_conservative_10852x4604_srgb16.tif"}},
    }
    lines = report_builder.conclusions(manifest)
    assert lines[-3].startswith("Final 50 MP image: sr_conservative (first eligible output")
    assert "`final_scene_sr_conservative_10852x4604_srgb16.tif`" in lines[-3]


def test_ranking_needs_two_measured_outputs():
    manifest = colab_manifest()
    manifest["baselines"] = {"lanczos": manifest["baselines"]["lanczos"]}
    manifest["super_resolution"] = {}
    assert not any("best first" in line for line in report_builder.conclusions(manifest))
