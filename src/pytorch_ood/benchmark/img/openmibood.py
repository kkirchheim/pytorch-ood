"""
Medical imaging OOD benchmarks from OpenMIBOOD (CVPR 2025).

:see Paper: `OpenMIBOOD <https://arxiv.org/abs/2503.16247>`__
:see Setup: https://github.com/remic-othr/OpenMIBOOD
"""

import os
from os.path import join
from typing import Any, Callable, ClassVar, List, Optional

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.transforms import Compose

from pytorch_ood.benchmark import Benchmark
from pytorch_ood.dataset.img import ImageListDataset
from pytorch_ood.utils import ToRGB, ToUnknown

_IMGLIST_BASE = (
    "https://raw.githubusercontent.com/remic-othr/OpenMIBOOD/main/data/benchmark_imglist"
)


class _OpenMIBOODBase(Benchmark):
    """Shared structure for the three OpenMIBOOD benchmarks."""

    _dataset_subdir: ClassVar[str]
    _train_imglist: ClassVar[str]
    _test_imglist: ClassVar[str]
    _ood_imglists: ClassVar[List[str]]

    cs_id_names: ClassVar[List[str]]  #: covariate-shifted ID dataset names
    near_ood_names: ClassVar[List[str]]  #: near-OOD dataset names
    far_ood_names: ClassVar[List[str]]  #: far-OOD dataset names

    def __init__(
        self,
        root: str,
        transform: Callable,
        loader: Optional[Callable[[str], Any]] = None,
        download: bool = True,
    ) -> None:
        """
        :param root: directory containing the prepared OpenMIBOOD data for this benchmark
        :param transform: transform applied to each loaded image (after :class:`ToRGB`)
        :param loader: callable mapping a file path to an image; defaults to :func:`PIL.Image.open`.
            Required for benchmarks whose image format is not handled by PIL (e.g. NIfTI).
        :param download: if ``True``, download missing imglist files to ``root/imglists/<bench>/``.
            If ``False``, raise an error if any required file is missing. Defaults to ``True``.
        """
        self.transform = Compose([ToRGB(), transform])
        imglist_dir = os.path.join(root, "imglists", self._dataset_subdir)
        os.makedirs(imglist_dir, exist_ok=True)

        all_imglists = [self._train_imglist, self._test_imglist] + list(self._ood_imglists)
        for fname in all_imglists:
            fpath = os.path.join(imglist_dir, fname)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError(
                        f"Imglist file not found: {fpath}. "
                        "Pass download=True to fetch it automatically, "
                        f"or download manually from {_IMGLIST_BASE}/{self._dataset_subdir}/{fname}"
                    )
                url = f"{_IMGLIST_BASE}/{self._dataset_subdir}/{fname}"
                download_url(url, imglist_dir, filename=fname)

        bench_dir = imglist_dir

        def _load(imglist_name: str, ood: bool) -> ImageListDataset:
            target_transform = ToUnknown() if ood else None
            return ImageListDataset(
                root=root,
                imglist_path=join(bench_dir, imglist_name),
                transform=self.transform,
                target_transform=target_transform,
                loader=loader,
            )

        self.train_in = _load(self._train_imglist, ood=False)
        self.test_in = _load(self._test_imglist, ood=False)
        self.test_oods = [_load(f, ood=True) for f in self._ood_imglists]

        self.ood_names: List[str] = (
            list(self.cs_id_names) + list(self.near_ood_names) + list(self.far_ood_names)
        )
        if len(self.ood_names) != len(self.test_oods):
            raise RuntimeError(
                f"Internal mismatch: {len(self.ood_names)} ood_names vs "
                f"{len(self.test_oods)} test datasets"
            )

    def train_set(self) -> Dataset:
        return self.train_in

    def test_sets(self, known: bool = True, unknown: bool = True) -> List[Dataset]:
        if known and unknown:
            return [self.test_in + other for other in self.test_oods]
        if known and not unknown:
            return [self.train_in]
        if not known and unknown:
            return self.test_oods
        raise ValueError("At least one of `known` or `unknown` must be True")


class MIDOG_OpenMIBOOD(_OpenMIBOODBase):
    """
    Replicates the MIDOG benchmark proposed in
    *OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection*.
    Images are 50x50 TIFF patches; the in-distribution task is 3-class mitosis
    classification on Domain 1a.

    Requires data prepared following the OpenMIBOOD setup guide. ``root`` should
    point at the directory whose subfolders match the bundled imglist paths
    (e.g. ``1a/017/017_342_0.tiff``).

    Covariate-shifted ID datasets:

     * ``midog_csid_1b`` — Domain 1b (different scanner, same task)
     * ``midog_csid_1c`` — Domain 1c (different scanner, same task)

    Near-OOD datasets (other scanner/staining domains):

     * ``midog_2``, ``midog_3``, ``midog_4``, ``midog_5``, ``midog_6a``, ``midog_6b``, ``midog_7``

    Far-OOD datasets (different cytology):

     * ``midog_ccagt`` — cervical cells (CCAgT)
     * ``midog_fnac2019`` — fine-needle aspirate cytology (FNAC 2019)

    :see Paper: `OpenMIBOOD <https://arxiv.org/abs/2503.16247>`__
    :see Setup: https://github.com/remic-othr/OpenMIBOOD
    """

    _dataset_subdir = "midog"
    _train_imglist = "train_midog.txt"
    _test_imglist = "test_midog.txt"
    _ood_imglists = [
        "test_midog_1b.txt",
        "test_midog_1c.txt",
        "test_midog_2.txt",
        "test_midog_3.txt",
        "test_midog_4.txt",
        "test_midog_5.txt",
        "test_midog_6a.txt",
        "test_midog_6b.txt",
        "test_midog_7.txt",
        "test_midog_ccagt.txt",
        "test_midog_fnac2019.txt",
    ]
    cs_id_names = ["midog_csid_1b", "midog_csid_1c"]
    near_ood_names = [
        "midog_2",
        "midog_3",
        "midog_4",
        "midog_5",
        "midog_6a",
        "midog_6b",
        "midog_7",
    ]
    far_ood_names = ["midog_ccagt", "midog_fnac2019"]


class PhaKIR_OpenMIBOOD(_OpenMIBOODBase):
    """
    Replicates the PhaKIR benchmark proposed in
    *OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection*.
    Images are PNG video frames; the in-distribution task is 7-class surgical-phase
    classification on PhaKIR videos 02-04 and 07 (Video 01 is held out for testing).

    Requires data prepared following the OpenMIBOOD setup guide. ``root`` should
    point at the directory whose subfolders match the bundled imglist paths
    (e.g. ``Video_02/Video_02_Frames/frame_0_19_0.png``).

    Covariate-shifted ID datasets:

     * ``phakir_medium_smoke`` — same procedure with medium smoke artifacts
     * ``phakir_heavy_smoke`` — same procedure with heavy smoke artifacts

    Near-OOD datasets (other laparoscopic surgery videos):

     * ``phakir_cholec`` — Cholec80
     * ``phakir_endovis2015`` — EndoVis 2015
     * ``phakir_endovis2018`` — EndoVis 2018

    Far-OOD datasets (different surgical/clinical domains):

     * ``phakir_kvasir`` — Kvasir-SEG (gastrointestinal endoscopy)
     * ``phakir_cataracts`` — CATARACTS (ophthalmic surgery)

    :see Paper: `OpenMIBOOD <https://arxiv.org/abs/2503.16247>`__
    :see Setup: https://github.com/remic-othr/OpenMIBOOD
    """

    _dataset_subdir = "phakir"
    _train_imglist = "train_phakir.txt"
    _test_imglist = "test_phakir.txt"
    _ood_imglists = [
        "test_phakir_medium_smoke_csid.txt",
        "test_phakir_heavy_smoke_csid.txt",
        "test_phakir_cholec_near.txt",
        "test_phakir_endovis2015_near.txt",
        "test_phakir_endovis2018_near.txt",
        "test_phakir_kvasir_far.txt",
        "test_phakir_cataracts_far.txt",
    ]
    cs_id_names = ["phakir_medium_smoke", "phakir_heavy_smoke"]
    near_ood_names = ["phakir_cholec", "phakir_endovis2015", "phakir_endovis2018"]
    far_ood_names = ["phakir_kvasir", "phakir_cataracts"]


class OASIS3_OpenMIBOOD(_OpenMIBOODBase):
    """
    Replicates the OASIS-3 benchmark proposed in
    *OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection*.
    Images are NIfTI (``.nii.gz``) 3D volumes (skull-stripped, resampled); the
    in-distribution task is 2-class classification on T1w scans.

    Requires data prepared following the OpenMIBOOD setup guide. ``root`` should
    point at the directory whose subfolders match the bundled imglist paths
    (e.g. ``OASIS3/OAS30704/.../sub-OAS30704_..._T1w_resampled_skull_stripped.nii.gz``).

    .. note::

       NIfTI files are not handled by ``PIL.Image.open``. You must supply a
       ``loader`` callable that maps a file path to an image (typically a 2D
       slice extracted from the 3D volume). For example::

           import nibabel as nib
           from PIL import Image

           def load_central_slice(path):
               vol = nib.load(path).get_fdata()
               sl = vol[:, :, vol.shape[2] // 2]
               sl = (255 * (sl - sl.min()) / max(sl.ptp(), 1e-8)).astype("uint8")
               return Image.fromarray(sl)

           bench = OASIS3_OpenMIBOOD(root, transform=t, loader=load_central_slice)

    Covariate-shifted ID datasets:

     * ``oasis3_scanner`` — Siemens MAGNETOM Vision scanner T1w
     * ``oasis3_t2w`` — T2-weighted modality

    Near-OOD datasets (other brain MRI):

     * ``oasis3_atlas`` — ATLAS R2.0 (stroke lesions)
     * ``oasis3_brats`` — BraTS 2023 glioma
     * ``oasis3_ct`` — OASIS-3 CT

    Far-OOD datasets (other body regions):

     * ``oasis3_heart`` — MSD Task02 Heart
     * ``oasis3_chaos_inPhase`` — CHAOS abdominal MRI (in-phase)

    :see Paper: `OpenMIBOOD <https://arxiv.org/abs/2503.16247>`__
    :see Setup: https://github.com/remic-othr/OpenMIBOOD
    """

    _dataset_subdir = "oasis3"
    _train_imglist = "train_oasis3.txt"
    _test_imglist = "test_oasis3.txt"
    _ood_imglists = [
        "test_oasis3_scanner_csid.txt",
        "test_oasis3_t2w_csid.txt",
        "test_oasis3_atlas_near.txt",
        "test_oasis3_brats_near.txt",
        "test_oasis3_ct_near.txt",
        "test_oasis3_heart_far.txt",
        "test_oasis3_chaos_inPhase_far.txt",
    ]
    cs_id_names = ["oasis3_scanner", "oasis3_t2w"]
    near_ood_names = ["oasis3_atlas", "oasis3_brats", "oasis3_ct"]
    far_ood_names = ["oasis3_heart", "oasis3_chaos_inPhase"]
