"""
Medical imaging OOD benchmarks from OpenMIBOOD (CVPR 2025).

See the `OpenMIBOOD setup guide <https://github.com/remic-othr/OpenMIBOOD>`__.
"""

import logging
import os
from os.path import join
from typing import Any, Callable, ClassVar, List, Optional

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision.transforms import Compose

from pytorch_ood.api import BenchmarkInfo, Paper, Task
from pytorch_ood.benchmark import Benchmark
from pytorch_ood.dataset.img import ImageListDataset
from pytorch_ood.dataset.img.openmibood import (
    ATLAS,
    CATARACTS,
    CHAOS,
    FNAC2019,
    MIDOG,
    OASIS3,
    BraTS,
    CCAgT,
    Cholec80,
    EndoVis2018,
    KvasirSEG,
    PhaKIR,
    Task02Heart,
)
from pytorch_ood.utils import ToRGB, ToUnknown

log = logging.getLogger(__name__)

_IMGLIST_BASE = (
    "https://raw.githubusercontent.com/remic-othr/OpenMIBOOD/main/data/benchmark_imglist"
)


class _OpenMIBOODBase(Benchmark):
    """Shared structure for the three OpenMIBOOD benchmarks."""

    _dataset_subdir: ClassVar[str]
    _check_dir: ClassVar[str]  # subdirectory expected directly below the data root
    _train_imglist: ClassVar[str]
    _test_imglist: ClassVar[str]
    _ood_imglists: ClassVar[List[str]]
    _datasets: ClassVar[
        List[Any]
    ] = []  # constituent datasets that can be downloaded automatically

    cs_id_names: ClassVar[List[str]]  #: covariate-shifted ID dataset names
    near_ood_names: ClassVar[List[str]]  #: near-OOD dataset names
    far_ood_names: ClassVar[List[str]]  #: far-OOD dataset names

    def __init__(
        self,
        root: str,
        transform: Callable,
        loader: Optional[Callable[[str], Any]] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: directory containing the prepared OpenMIBOOD data for this benchmark
        :param transform: transform applied to each loaded image (after :class:`~pytorch_ood.utils.ToRGB`)
        :param loader: callable mapping a file path to an image; defaults to :func:`PIL.Image.open`.
            Required for benchmarks whose image format is not handled by PIL (e.g. NIfTI);
            :class:`OASIS3_OpenMIBOOD` cannot load its images without it.
        :param download: if ``True``, download missing imglist files to ``root/imglists/<bench>/`` and
            download the constituent datasets that are publicly available. Datasets that can not be
            downloaded automatically have to be prepared manually (see the class documentation). If
            ``False``, raise an error if any required imglist file is missing. Defaults to ``False``.
        :raises FileNotFoundError: if the images referenced by the imglists can not be found below ``root``
        """
        # the data may also be located in a subdirectory of root, e.g., root/midog
        if not os.path.isdir(join(root, self._check_dir)):
            for sub in [self._dataset_subdir, f"{self._dataset_subdir}_extracted"]:
                if os.path.isdir(join(root, sub, self._check_dir)):
                    root = join(root, sub)
                    break

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

        if download:
            for dataset_cls in self._datasets:
                try:
                    dataset_cls(root=root, download=True)
                except Exception as e:
                    log.warning(f"Could not set up {dataset_cls.__name__}: {e}")

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

        if self.train_in.files and not os.path.exists(self.train_in.files[0]):
            raise FileNotFoundError(
                f"Image '{self.train_in.files[0]}' not found. 'root' must contain the prepared data, "
                "with subfolders matching the paths in the imglists."
            )

        self.ood_names: List[str] = (
            list(self.cs_id_names) + list(self.near_ood_names) + list(self.far_ood_names)
        )
        if len(self.ood_names) != len(self.test_oods):
            raise RuntimeError(
                f"Internal mismatch: {len(self.ood_names)} ood_names vs "
                f"{len(self.test_oods)} test datasets"
            )

        missing = [
            name
            for name, dataset in zip(self.ood_names, self.test_oods)
            if dataset.files and not os.path.exists(dataset.files[0])
        ]
        if missing:
            log.warning(
                f"Data for the test sets {missing} was not found below '{root}'. Evaluating on them "
                f"will fail. Some datasets can not be downloaded automatically, see the documentation "
                f"of {self.__class__.__name__}."
            )

    def train_set(self) -> Dataset:
        return self.train_in

    def test_sets(self, known: bool = True, unknown: bool = True) -> List[Dataset]:
        """
        List of the different test datasets.

        :param known: include ID
        :param unknown: include OOD
        :return: with ``known`` and ``unknown``, one dataset per entry of ``ood_names`` that
            combines the ID test set with that OOD set. With ``unknown`` only, the OOD sets in the
            order of ``ood_names``. With ``known`` only, the ID test set, once.
        :raises ValueError: if both ``known`` and ``unknown`` are false
        """
        if known and unknown:
            return [self.test_in + other for other in self.test_oods]
        if known and not unknown:
            return [self.test_in]
        if not known and unknown:
            return self.test_oods
        raise ValueError("At least one of `known` or `unknown` must be True")


class MIDOG_OpenMIBOOD(_OpenMIBOODBase):
    """
    Replicates the MIDOG benchmark proposed in
    *OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection*.

    **What the data is.** The MIDOG++ dataset contains 503 microscope scans (whole-slide images) of
    tissue samples stained with Hematoxylin & Eosin, covering seven cancer types from humans and
    dogs. Pathologists annotated *mitotic figures* (cells that are dividing) and *imposters*
    (cells that look similar but are not dividing).

    **In-distribution task.** 3-class classification of 50x50 pixel patches from domain 1a (human
    breast cancer, one scanner): mitotic figure, imposter, or background (a patch shifted 100 px
    next to an annotation that contains neither).

    ``ood_names`` lists the covariate-shifted ID, near-OOD and far-OOD sets in this order (also
    available as the class attributes ``cs_id_names``, ``near_ood_names`` and ``far_ood_names``).
    The covariate-shifted ID sets are loaded with :class:`~pytorch_ood.utils.ToUnknown`, i.e.,
    they are labelled as OOD (``-1``) in :meth:`~pytorch_ood.benchmark.Benchmark.test_sets`.

    Covariate-shifted ID (same task, different scanner):

    * midog_csid_1b, midog_csid_1c --- domains 1b and 1c of :class:`~pytorch_ood.dataset.img.MIDOG`

    Near-OOD (other cancer types and species, often also other scanners or labs):

    * midog_2, midog_3, midog_4, midog_5, midog_6a, midog_6b, midog_7 --- the remaining domains of
      :class:`~pytorch_ood.dataset.img.MIDOG`

    Far-OOD (cell images from other medical applications):

    * midog_ccagt --- cervical cancer cells, :class:`~pytorch_ood.dataset.img.CCAgT`
    * midog_fnac2019 --- breast fine-needle aspiration cytology,
      :class:`~pytorch_ood.dataset.img.FNAC2019`

    **Data preparation.** With ``download=True``, the imglists are downloaded to
    ``<root>/imglists/midog/`` and :class:`~pytorch_ood.dataset.img.MIDOG` (all domains) and
    :class:`~pytorch_ood.dataset.img.CCAgT` are downloaded and cut into patches automatically.
    This downloads all 503 slide images one after another (each is deleted after its patches are
    extracted), which takes a long time.

    .. warning::

       :class:`~pytorch_ood.dataset.img.FNAC2019` can not be downloaded automatically. Without
       it, the ``midog_fnac2019`` test set is missing. See its documentation for instructions.

    ``root`` must be the directory whose subfolders match the paths in the imglists
    (e.g. ``1a/017/017_342_0.tiff``).
    """

    info = BenchmarkInfo(
        paper=Paper(
            title="OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection",
            venue="CVPR",
            year=2025,
            url="https://arxiv.org/abs/2503.16247",
            code="https://github.com/remic-othr/OpenMIBOOD",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    _dataset_subdir = "midog"
    _check_dir = "1a"
    _datasets = [MIDOG, CCAgT, FNAC2019]
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

    **What the data is.** Frames (PNG images) from videos of laparoscopic gallbladder removals
    (cholecystectomies) from the PhaKIR challenge.

    **In-distribution task.** 7-class classification of which surgical instrument is visible in a
    frame, including a "no instrument" class. Only frames with at most one instrument and no
    visible smoke are used. Videos 02-05 and 07 are used for training and validation, video 01 for
    testing.

    ``ood_names`` lists the covariate-shifted ID, near-OOD and far-OOD sets in this order (also
    available as the class attributes ``cs_id_names``, ``near_ood_names`` and ``far_ood_names``).
    The covariate-shifted ID sets are labelled as OOD (``-1``) in
    :meth:`~pytorch_ood.benchmark.Benchmark.test_sets`.

    Covariate-shifted ID (same videos, but smoke from tissue coagulation):

    * phakir_medium_smoke, phakir_heavy_smoke --- part of :class:`~pytorch_ood.dataset.img.PhaKIR`

    Near-OOD (other surgeries and instruments):

    * phakir_cholec --- other gallbladder surgeries, :class:`~pytorch_ood.dataset.img.Cholec80`
    * phakir_endovis2015 --- colorectal surgery (EndoVis 2015), no dataset class, see below
    * phakir_endovis2018 --- robotic kidney surgery on pigs,
      :class:`~pytorch_ood.dataset.img.EndoVis2018`

    Far-OOD (other medical fields):

    * phakir_kvasir --- polyps in the bowel, :class:`~pytorch_ood.dataset.img.KvasirSEG`
    * phakir_cataracts --- eye surgery, :class:`~pytorch_ood.dataset.img.CATARACTS`

    **Data preparation.** With ``download=True``, the imglists are downloaded to
    ``<root>/imglists/phakir/`` and :class:`~pytorch_ood.dataset.img.Cholec80`,
    :class:`~pytorch_ood.dataset.img.KvasirSEG` and :class:`~pytorch_ood.dataset.img.CATARACTS`
    are downloaded automatically.

    .. warning::

       The in-distribution data (:class:`~pytorch_ood.dataset.img.PhaKIR`) and
       :class:`~pytorch_ood.dataset.img.EndoVis2018` require registration and can not be
       downloaded automatically. See their documentation for instructions.

       EndoVis 2015 is no longer available: the server hosting it
       (``opencas.webarchiv.kit.edu``) does not exist anymore. Unless you already have a copy,
       ``phakir_endovis2015`` can not be evaluated.

    ``root`` must be the directory whose subfolders match the paths in the imglists
    (e.g. ``Video_02/Video_02_Frames/frame_0_19_0.png``).
    """

    info = BenchmarkInfo(
        paper=Paper(
            title="OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection",
            venue="CVPR",
            year=2025,
            url="https://arxiv.org/abs/2503.16247",
            code="https://github.com/remic-othr/OpenMIBOOD",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    _dataset_subdir = "phakir"
    _check_dir = "Video_02"
    # EndoVis 2015 is no longer publicly hosted and has to be prepared manually
    _datasets = [PhaKIR, Cholec80, EndoVis2018, KvasirSEG, CATARACTS]
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

    **What the data is.** 3D brain MRI scans from the OASIS-3 study, stored as NIfTI
    (``.nii.gz``) volumes. All scans are resampled to 1 mm voxels and skull-stripped (the skull is
    removed with HD-BET).

    **In-distribution task.** 2-class classification of T1-weighted (T1w) MRI scans into
    cognitively normal (CN) subjects and subjects with Alzheimer's disease (AD).

    ``ood_names`` lists the covariate-shifted ID, near-OOD and far-OOD sets in this order (also
    available as the class attributes ``cs_id_names``, ``near_ood_names`` and ``far_ood_names``).
    The covariate-shifted ID sets are labelled as OOD (``-1``) in
    :meth:`~pytorch_ood.benchmark.Benchmark.test_sets`.

    Covariate-shifted ID (same subjects and task, different acquisition):

    * oasis3_scanner --- T1w scans from a different MRI scanner (Siemens MAGNETOM Vision), which
      were held out from training, :class:`~pytorch_ood.dataset.img.OASIS3`
    * oasis3_t2w --- T2-weighted scans instead of T1w, :class:`~pytorch_ood.dataset.img.OASIS3`

    Near-OOD (other brain conditions or imaging methods):

    * oasis3_atlas --- brains with stroke lesions, :class:`~pytorch_ood.dataset.img.ATLAS`
    * oasis3_brats --- brains with large tumors (gliomas), :class:`~pytorch_ood.dataset.img.BraTS`
    * oasis3_ct --- CT instead of MRI scans of the OASIS-3 subjects,
      :class:`~pytorch_ood.dataset.img.OASIS3`

    Far-OOD (other body regions):

    * oasis3_heart --- heart MRI, :class:`~pytorch_ood.dataset.img.Task02Heart`
    * oasis3_chaos_inPhase --- abdominal MRI, :class:`~pytorch_ood.dataset.img.CHAOS`

    **Data preparation.** With ``download=True``, the imglists are downloaded to
    ``<root>/imglists/oasis3/`` and :class:`~pytorch_ood.dataset.img.Task02Heart` and
    :class:`~pytorch_ood.dataset.img.CHAOS` are downloaded automatically. This requires the
    additional packages ``gdown`` (for Task02Heart) and ``SimpleITK`` (to convert CHAOS from DICOM
    to NIfTI)::

        pip install gdown SimpleITK

    .. warning::

       :class:`~pytorch_ood.dataset.img.OASIS3` (the in-distribution data),
       :class:`~pytorch_ood.dataset.img.ATLAS` and :class:`~pytorch_ood.dataset.img.BraTS` require
       registration and can not be downloaded automatically. OASIS3 and ATLAS must also be
       resampled and skull-stripped with
       `HD-BET <https://github.com/MIC-DKFZ/HD-BET>`__ (version 2.0.1 to reproduce the paper)
       using the scripts from the `OpenMIBOOD repository
       <https://github.com/remic-othr/OpenMIBOOD/tree/main/scripts/download/OpenMIBOOD>`__. See
       their documentation for instructions.

    ``root`` must be the directory whose subfolders match the paths in the imglists
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
    """

    info = BenchmarkInfo(
        paper=Paper(
            title="OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection",
            venue="CVPR",
            year=2025,
            url="https://arxiv.org/abs/2503.16247",
            code="https://github.com/remic-othr/OpenMIBOOD",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    _dataset_subdir = "oasis3"
    _check_dir = "OASIS3"
    _datasets = [OASIS3, ATLAS, BraTS, Task02Heart, CHAOS]
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
