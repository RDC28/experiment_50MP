"""Full-scale experimental preset, with declared modest synthetic priors."""

from prd_v5.analytic import AnalyticISP
from prd_v5.config import AnalyticRunConfig
from prd_v5.optical_effects import OpticalEffects
from prd_v5.optics import Distortion, Vignetting
from prd_v5.sensor import SensorEffects


def full_quality_config():
    """Enable implemented modules; maximum coverage does not mean maximum damage.

    No private EXIF fabrication or model licence acceptance is implied. Missing
    research checkpoints remain visible in dispatch evidence. Large runs belong
    in Colab; library callers retain the conservative AnalyticRunConfig defaults.
    """
    return AnalyticRunConfig(
        resampling="LANCZOS",
        target_megapixels=50,
        assume_unprofiled_srgb=True,
        max_decode_bytes=1024**3,
        max_develop_bytes=8 * 1024**3,
        accept_experimental_optics=True,
        isp=AnalyticISP(hdr_max_radiance=16),
        sensor_effects=SensorEffects(
            prnu_sigma=0.005,
            dsnu_e=0.3,
            row_fpn_e=0.5,
            column_fpn_e=0.5,
            dark_current_e_per_s=0.05,
            hot_pixel_fraction=2e-6,
            dead_pixel_fraction=2e-8,
            nonlinearity_compression=0.005,
        ),
        vignetting=Vignetting(0.2),
        distortion=Distortion(k1=0.008, k2=0.002, p1=0.0001, p2=-0.0001),
        optical_effects=OpticalEffects(
            lateral_ca_px=0.5,
            axial_ca_sigma_px=0.2,
            psf_center_sigma_px=0.15,
            psf_edge_sigma_px=0.3,
            diffraction=True,
            defocus_radius_px=0.5,
            motion_length_px=0.3,
        ),
    )
