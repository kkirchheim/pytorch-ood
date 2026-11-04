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
from pytorch_ood.utils import ToRGB, ToUnknown

from . import _openmibood_download as data

log = logging.getLogger(__name__)

_IMGLIST_BASE = (
    "https://raw.githubusercontent.com/remic-othr/OpenMIBOOD/main/data/benchmark_imglist"
)


class _OpenMIBOODBase(Benchmark):
    """Shared structure for the three OpenMIBOOD benchmarks."""

    _dataset_subdir: ClassVar[str]
    _train_imglist: ClassVar[str]
    _val_imglist: ClassVar[str]
    _test_imglist: ClassVar[str]
    _ood_imglists: ClassVar[List[str]]
    _sources: ClassVar[List[data._Source]]  # constituent datasets

    cs_id_names: ClassVar[List[str]]  #: covariate-shifted ID dataset names
    near_ood_names: ClassVar[List[str]]  #: near-OOD dataset names
    far_ood_names: ClassVar[List[str]]  #: far-OOD dataset names

    def __init__(
        self,
        root: str,
        transform: Callable,
        loader: Optional[Callable[[str], Any]] = None,
        download: bool = False,
        skip_missing: bool = False,
    ) -> None:
        """
        :param root: directory containing the prepared OpenMIBOOD data for this benchmark, with
            subfolders matching the paths in the imglists
        :param transform: transform applied to each loaded image (after :class:`~pytorch_ood.utils.ToRGB`)
        :param loader: callable mapping a file path to an image; defaults to :func:`PIL.Image.open`.
            Required for benchmarks whose image format is not handled by PIL (e.g. NIfTI);
            :class:`OASIS3_OpenMIBOOD` cannot load its images without it.
        :param download: if ``True``, download missing imglist files to ``root/imglists/<bench>/``
            and download the constituent datasets that are publicly available and not yet present.
            Datasets that require registration have to be prepared manually (see the class
            documentation).
        :param skip_missing: if ``True``, leave out OOD test sets whose data is missing instead of
            raising an error. They are then also missing from ``ood_names`` and the results of
            :meth:`~pytorch_ood.benchmark.Benchmark.evaluate`, so the results are not comparable to
            the paper.
        :raises RuntimeError: if an imglist file is missing and ``download`` is ``False``
        :raises FileNotFoundError: if the in-distribution data is missing, or the data of an OOD
            test set is missing and ``skip_missing`` is ``False``
        """
        self.transform = Compose([ToRGB(), transform])
        imglist_dir = os.path.join(root, "imglists", self._dataset_subdir)
        os.makedirs(imglist_dir, exist_ok=True)

        all_imglists = [self._train_imglist, self._val_imglist, self._test_imglist] + list(
            self._ood_imglists
        )
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
            for source in self._sources:
                if source.download is not None and not source.is_present(root):
                    log.info(f"Downloading {source.name}")
                    source.download(root)

        def _load(imglist_name: str, ood: bool) -> ImageListDataset:
            target_transform = ToUnknown() if ood else None
            return ImageListDataset(
                root=root,
                imglist_path=join(imglist_dir, imglist_name),
                transform=self.transform,
                target_transform=target_transform,
                loader=loader,
            )

        def _missing(dataset: ImageListDataset) -> bool:
            return bool(dataset.files) and not os.path.exists(dataset.files[0])

        missing_sources = [s.name for s in self._sources if not s.is_present(root)]
        hint = (
            f"Datasets not found below '{root}': {missing_sources}. See the documentation of "
            f"{self.__class__.__name__} for how to prepare them."
        )

        self.train_in = _load(self._train_imglist, ood=False)
        self.val_in = _load(self._val_imglist, ood=False)
        self.test_in = _load(self._test_imglist, ood=False)
        if any(_missing(d) for d in (self.train_in, self.val_in, self.test_in)):
            raise FileNotFoundError(f"The in-distribution data is missing. {hint}")

        ood_names = list(self.cs_id_names) + list(self.near_ood_names) + list(self.far_ood_names)
        test_oods = [_load(f, ood=True) for f in self._ood_imglists]
        if len(ood_names) != len(test_oods):
            raise RuntimeError(
                f"Internal mismatch: {len(ood_names)} ood_names vs {len(test_oods)} test datasets"
            )

        missing = [name for name, d in zip(ood_names, test_oods) if _missing(d)]
        if missing and not skip_missing:
            raise FileNotFoundError(
                f"The data of the test sets {missing} is missing. Pass skip_missing=True to "
                f"evaluate without them. {hint}"
            )
        if missing:
            log.warning(f"Leaving out the test sets {missing}, their data is missing. {hint}")

        # names of the OOD test sets, in the order of test_sets()
        self.ood_names: List[str] = [n for n in ood_names if n not in missing]
        self.test_oods = [d for n, d in zip(ood_names, test_oods) if n not in missing]

    def train_set(self) -> Dataset:
        return self.train_in

    def val_set(self) -> Dataset:
        """
        In-distribution validation set, which OpenMIBOOD uses to tune hyperparameters and to fit
        some detectors.
        """
        return self.val_in

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

    * midog_csid_1b, midog_csid_1c --- domains 1b and 1c of MIDOG++

    Near-OOD (other cancer types and species, often also other scanners or labs):

    * midog_2, midog_3, midog_4, midog_5, midog_6a, midog_6b, midog_7 --- the remaining domains of
      MIDOG++

    Far-OOD (cell images from other medical applications):

    * midog_ccagt --- cervical cancer cells stained with the AgNOR technique
      (`CCAgT <https://data.mendeley.com/datasets/wg4bpm33hj/2>`__)
    * midog_fnac2019 --- breast fine-needle aspiration cytology (FNAC 2019)

    **Data preparation.** ``root`` must be the directory whose subfolders match the paths in the
    imglists (e.g. ``1a/017/017_342_0.tiff``). With ``download=True``, the imglists are downloaded
    to ``<root>/imglists/midog/``, and the following datasets are downloaded and cut into 50x50
    pixel patches as in the paper:

    * `MIDOG++ <https://doi.org/10.1038/s41597-023-02327-4>`__: the 503 slides (65 GB) are
      downloaded one after another from Figshare, and each is deleted after its patches are
      extracted. This takes a long time, but can be interrupted and resumed.
    * CCAgT: one patch centered on each annotated nucleus.

    .. warning::

       FNAC 2019 can not be downloaded automatically, because it is only shared via a OneDrive
       link. Without it, the ``midog_fnac2019`` test set is missing, see ``skip_missing``.

       1. Download the archive from https://1drv.ms/u/s!Al-T6d-_ENf6axsEbvhbEc2gUFs.
       2. Extract the patches with ``download_midog_fnac2019.py`` from the
          `OpenMIBOOD scripts <https://github.com/remic-othr/OpenMIBOOD/tree/main/scripts/download/OpenMIBOOD>`__ (requires ``opencv-python``).
       3. Place the result in ``<root>/far/fnac2019_crops/``.
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
    _sources = [data.MIDOG, data.CCAGT, data.FNAC2019]
    _train_imglist = "train_midog.txt"
    _val_imglist = "valid_midog.txt"
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

    * phakir_medium_smoke, phakir_heavy_smoke

    Near-OOD (other surgeries and instruments):

    * phakir_cholec --- other gallbladder surgeries (Cholec80)
    * phakir_endovis2015 --- colorectal surgery (EndoVis 2015)
    * phakir_endovis2018 --- robotic kidney surgery on pigs (EndoVis 2018)

    Far-OOD (other medical fields):

    * phakir_kvasir --- polyps in the bowel (`Kvasir-SEG <https://datasets.simula.no/kvasir-seg/>`__)
    * phakir_cataracts --- eye surgery (CATARACTS)

    **Data preparation.** ``root`` must be the directory whose subfolders match the paths in the
    imglists (e.g. ``Video_02/Video_02_Frames/frame_0_19_0.png``). With ``download=True``, the
    imglists are downloaded to ``<root>/imglists/phakir/``, and Kvasir-SEG as well as the versions
    of Cholec80 (cropped frames) and CATARACTS (cleaned subset) prepared by the OpenMIBOOD authors
    are downloaded automatically (about 50 GB).

    .. warning::

       The other datasets can not be downloaded automatically.

       * **PhaKIR** (in-distribution data): request access at
         https://doi.org/10.5281/zenodo.16753918, download ``phakir_openmibood.zip`` and extract
         it into ``<root>``, so that the ``Video_XX`` folders are directly below ``<root>``.
       * **EndoVis 2018**: register at
         https://endovissub2018-roboticscenesegmentation.grand-challenge.org/, join the challenge
         and download the test data. Copy ``test_data/seq_<i>/left_frames`` of sequences 1-4 to
         ``<root>/near/Endovis2018/seq_<i>/left_frames``, or run
         ``download_phakir_endovis2018.py`` from the `OpenMIBOOD scripts <https://github.com/remic-othr/OpenMIBOOD/tree/main/scripts/download/OpenMIBOOD>`__, which
         also adds the frame ``seq_2/left_frames/frame249.png`` that is missing from the current
         download. See the `instructions <https://github.com/remic-othr/OpenMIBOOD/tree/main/instructions/Endovis2018>`__.
       * **EndoVis 2015** is no longer available: the server hosting it
         (``opencas.webarchiv.kit.edu``) does not exist anymore. Unless you already have a copy
         in ``<root>/near/Endovis2015_cropped/``, ``phakir_endovis2015`` can not be evaluated,
         see ``skip_missing``.
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
    _sources = [
        data.PHAKIR,
        data.CHOLEC80,
        data.ENDOVIS2015,
        data.ENDOVIS2018,
        data.KVASIR,
        data.CATARACTS,
    ]
    _train_imglist = "train_phakir.txt"
    _val_imglist = "valid_phakir.txt"
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
      were held out from training
    * oasis3_t2w --- T2-weighted scans instead of T1w

    Near-OOD (other brain conditions or imaging methods):

    * oasis3_atlas --- brains with stroke lesions (ATLAS R2.0)
    * oasis3_brats --- brains with large tumors (gliomas, BraTS 2023)
    * oasis3_ct --- CT instead of MRI scans of the OASIS-3 subjects

    Far-OOD (other body regions):

    * oasis3_heart --- heart MRI (task 2 of the `Medical Segmentation Decathlon
      <http://medicaldecathlon.com/>`__)
    * oasis3_chaos_inPhase --- abdominal MRI (`CHAOS <https://zenodo.org/records/3431873>`__)

    **Data preparation.** With ``download=True``, the imglists are downloaded to
    ``<root>/imglists/oasis3/``, and the heart and CHAOS scans are downloaded automatically. The
    conversion of CHAOS from DICOM to NIfTI requires ``SimpleITK`` (``pip install SimpleITK``).

    .. warning::

       The other datasets require registration and can not be downloaded automatically. OASIS-3
       and ATLAS must also be resampled to 1 mm voxels and skull-stripped with
       `HD-BET <https://github.com/MIC-DKFZ/HD-BET>`__ (version 2.0.1 to reproduce the paper),
       using the `OpenMIBOOD scripts <https://github.com/remic-othr/OpenMIBOOD/tree/main/scripts/download/OpenMIBOOD>`__.

       * **OASIS-3** (in-distribution, covariate-shifted and CT data): request access to
         "OASIS-3: Longitudinal Multimodal Imaging" at
         https://sites.wustl.edu/oasisbrains/home/access/ and register a NITRC account with the
         same email. Download all sessions with scan types CT, T1w and T2w in NIfTI format,
         following the `instructions <https://github.com/remic-othr/OpenMIBOOD/tree/main/instructions/OASIS3>`__, and run ``download_oasis3.py``. The
         results must be placed in ``<root>/OASIS3/``.
       * **ATLAS R2.0**: agree to the terms at
         https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html, download and decrypt
         ``ATLAS_R2.0.tar.gz`` and run ``download_oasis3_atlas.py``. The results must be placed in
         ``<root>/near/ATLAS/``. See the `instructions <https://github.com/remic-othr/OpenMIBOOD/tree/main/instructions/ATLAS>`__.
       * **BraTS 2023**: register at https://www.synapse.org/Synapse:syn51156910/wiki/627000,
         download ``ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`` and extract it with
         ``download_oasis3_brats.py`` into ``<root>/near/BraTS2023-GLI/``. See the
         `instructions <https://github.com/remic-othr/OpenMIBOOD/tree/main/instructions/BraTS>`__.

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
    _sources = [data.OASIS3, data.ATLAS, data.BRATS, data.HEART, data.CHAOS]
    _train_imglist = "train_oasis3.txt"
    _val_imglist = "valid_oasis3.txt"
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
