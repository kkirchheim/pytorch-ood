"""
Datasets used in the OpenMIBOOD (CVPR 2025) medical imaging benchmarks.

:see Paper: `OpenMIBOOD <https://arxiv.org/abs/2503.16247>`__
:see Setup: https://github.com/remic-othr/OpenMIBOOD
"""

import functools
import json
import logging
import os
import pathlib
import shutil
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from os.path import exists, isdir, isfile, join
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url

log = logging.getLogger(__name__)


def _download_url_with_ssl_fallback(url: str, root: str, filename: str, md5: Optional[str] = None):
    """
    Download a file from a URL to a local destination, with automatic SSL fallback
    for servers that omit intermediate certificates in their TLS handshake (e.g. Simula).
    """
    os.makedirs(root, exist_ok=True)
    fpath = join(root, filename)

    if check_integrity(fpath, md5):
        log.debug(f"File {fpath} already exists and verified.")
        return fpath

    try:
        download_url(url, root, filename=filename, md5=md5)
    except Exception as e:
        log.warning(f"Standard download failed ({e}). Retrying with SSL fallback context...")
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as response, open(fpath, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        if md5 and not check_integrity(fpath, md5):
            os.remove(fpath)
            raise RuntimeError(f"Downloaded file {fpath} failed MD5 integrity check.")

    return fpath


def _validate_patch(img_dims, annotations, additional_bbox):
    """Ensure additional cropped bounding box does not overlap with existing annotations."""
    if (
        additional_bbox[0] < 0
        or additional_bbox[1] < 0
        or additional_bbox[2] >= img_dims[0]
        or additional_bbox[3] >= img_dims[1]
    ):
        return False

    for annotation in annotations:
        bbox = annotation["bbox"]
        if bbox[0] < additional_bbox[0] < bbox[2] and bbox[1] < additional_bbox[1] < bbox[3]:
            return False
        if bbox[0] < additional_bbox[2] < bbox[2] and bbox[1] < additional_bbox[3] < bbox[3]:
            return False

    return True


class OpenMIBOODDataset(VisionDataset):
    """
    Abstract Base Class for OpenMIBOOD individual datasets.
    Each dataset class implements `download=True` and its own `download()` function.
    """

    url: Optional[str] = None
    filename: Optional[str] = None
    tgz_md5: Optional[str] = None
    target_rel_dir: Optional[str] = None

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(OpenMIBOODDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                f"Dataset not found or corrupted in {self.basedir}. "
                "Use download=True to fetch it automatically, or place the data manually."
            )

        self.files = self._load_files()

    @property
    def basedir(self) -> str:
        if self.target_rel_dir:
            return join(self.root, self.target_rel_dir)
        return self.root

    def _check_integrity(self) -> bool:
        if isdir(self.basedir) and len(os.listdir(self.basedir)) > 0:
            return True
        archive_path = join(self.root, self.filename) if self.filename else None
        if archive_path and isfile(archive_path):
            return check_integrity(archive_path, self.tgz_md5)
        return False

    def _load_files(self) -> List[str]:
        if not isdir(self.basedir):
            return []
        files = []
        for r, _, fnames in os.walk(self.basedir):
            for f in sorted(fnames):
                if not f.startswith(".") and f.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".nii.gz")
                ):
                    files.append(join(r, f))
        return files

    def download(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.files[index]
        target = -1

        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# =====================================================================
# PhaKIR Benchmark Datasets
# =====================================================================


class KvasirSEG(OpenMIBOODDataset):
    """
    Kvasir-SEG dataset for gastrointestinal polyp images (Far-OOD for PhaKIR).
    Contains 1,000 polyp images (900 used in PhaKIR benchmark split).

    :see Website: `Simula Kvasir-SEG <https://datasets.simula.no/kvasir-seg/>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1911.07069>`__
    """

    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    filename = "kvasir-seg.zip"
    tgz_md5 = "6323d9094df93b35d43069a566ee1ca3"
    target_rel_dir = "far/kvasir-seg"

    def download(self) -> None:
        dest_dir = join(self.root, self.target_rel_dir)
        if isdir(dest_dir) and len(os.listdir(dest_dir)) > 0:
            return

        archive_path = _download_url_with_ssl_fallback(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

        log.info(f"Extracting Kvasir-SEG -> {dest_dir}")
        tmp_extract = join(self.root, "_tmp_kvasir")
        os.makedirs(tmp_extract, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmp_extract)

        images_src = join(tmp_extract, "Kvasir-SEG", "images")
        if not isdir(images_src):
            images_src = join(tmp_extract, "images")

        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        if isdir(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.move(images_src, dest_dir)
        shutil.rmtree(tmp_extract, ignore_errors=True)


class CATARACTS(OpenMIBOODDataset):
    """
    CATARACTS surgical video dataset (Far-OOD for PhaKIR).
    Contains video frames from ophthalmic surgery.

    :see Zenodo: `CATARACTS Cleaned Subset <https://doi.org/10.5281/zenodo.14924735>`__
    """

    url = "https://zenodo.org/records/14924735/files/CATARACTS.zip?download=1"
    filename = "CATARACTS.zip"
    target_rel_dir = "far/CATARACTS"

    def download(self) -> None:
        dest_dir = join(self.root, self.target_rel_dir)
        if isdir(dest_dir) and len(os.listdir(dest_dir)) > 0:
            return

        archive_path = _download_url_with_ssl_fallback(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

        log.info(f"Extracting CATARACTS -> {dest_dir}")
        extract_parent = join(self.root, "far")
        os.makedirs(extract_parent, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_parent)


class Cholec80(OpenMIBOODDataset):
    """
    Cholec80 laparoscopic cholecystectomy cropped frames (Near-OOD for PhaKIR).

    :see Zenodo: `Cholec80 Cropped <https://doi.org/10.5281/zenodo.14921670>`__
    """

    url = "https://zenodo.org/records/14921670/files/Cholec80_cropped.zip?download=1"
    filename = "Cholec80_cropped.zip"
    target_rel_dir = "near/Cholec80_cropped"

    def download(self) -> None:
        dest_dir = join(self.root, self.target_rel_dir)
        if isdir(dest_dir) and len(os.listdir(dest_dir)) > 0:
            return

        archive_path = _download_url_with_ssl_fallback(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

        log.info(f"Extracting Cholec80 -> {dest_dir}")
        extract_parent = join(self.root, "near")
        os.makedirs(extract_parent, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_parent)


class PhaKIR(OpenMIBOODDataset):
    """
    In-distribution laparoscopic surgical instrument classification dataset (6 classes).
    
    .. warning::
        Automatic download of PhaKIR is not possible because the dataset has restricted access.
        Please request access at https://doi.org/10.5281/zenodo.16753918.
    """

    target_rel_dir = "Video_02"

    def download(self) -> None:
        if isdir(self.basedir) and len(os.listdir(self.basedir)) > 0:
            return

        raise RuntimeError(
            "Automatic download of PhaKIR is not possible because the dataset has restricted access.\n"
            "Please request access at https://doi.org/10.5281/zenodo.16753918.\n"
            "Once approved, place the extracted Video_XX folders under your benchmark root."
        )


class EndoVis2018(OpenMIBOODDataset):
    """
    EndoVis 2018 robotic surgery dataset (Near-OOD for PhaKIR).
    
    .. warning::
        Automatic download of EndoVis 2018 is not possible.
        Please register at https://endovissub2018-roboticscenesegmentation.grand-challenge.org/
        and download test sequences 1-4.
    """

    target_rel_dir = "near/Endovis2018"

    def download(self) -> None:
        if isdir(self.basedir) and len(os.listdir(self.basedir)) > 0:
            return

        raise RuntimeError(
            "Automatic download of EndoVis 2018 is not possible.\n"
            "Please register at https://endovissub2018-roboticscenesegmentation.grand-challenge.org/ "
            "and download test sequences 1-4 into near/Endovis2018/."
        )


# =====================================================================
# MIDOG Benchmark Datasets
# =====================================================================


class MIDOG(OpenMIBOODDataset):
    """
    MIDOG in-distribution mitosis dataset (3 classes: Mitosis, Hard Negative, Non-mitosis).
    Downloads WSI slides from Figshare and extracts 50x50 TIFF patches.

    :see Paper: `MIDOG++ <https://doi.org/10.1038/s41597-023-02327-4>`__
    """

    target_rel_dir = "1a"

    _FIGSHARE_FILES = {'40282102': '002.tiff', '40282099': '001.tiff', '40282096': '016.tiff', '40282105': '008.tiff', '40282111': '003.tiff', '40282108': '010.tiff', '40282114': '014.tiff', '40282132': '012.tiff', '40282129': '011.tiff', '40282126': '006.tiff', '40282117': '013.tiff', '40282123': '015.tiff', '40282135': '007.tiff', '40282120': '005.tiff', '40282138': '004.tiff', '40282141': '009.tiff', '40282144': '018.tiff', '40282147': '022.tiff', '40282153': '017.tiff', '40282150': '020.tiff', '40282162': '021.tiff', '40282156': '026.tiff', '40282159': '023.tiff', '40282165': '028.tiff', '40282180': '019.tiff', '40282171': '024.tiff', '40282186': '034.tiff', '40282183': '030.tiff', '40282168': '031.tiff', '40282174': '027.tiff', '40282177': '025.tiff', '40282189': '029.tiff', '40282192': '038.tiff', '40282195': '036.tiff', '40282198': '032.tiff', '40282204': '037.tiff', '40282207': '033.tiff', '40282201': '035.tiff', '40282210': '039.tiff', '40282213': '042.tiff', '40282225': '040.tiff', '40282216': '044.tiff', '40282219': '048.tiff', '40282222': '046.tiff', '40282234': '043.tiff', '40282231': '041.tiff', '40282228': '047.tiff', '40282237': '052.tiff', '40282240': '050.tiff', '40282243': '045.tiff', '40282246': '049.tiff', '40282252': '053.tiff', '40282249': '055.tiff', '40282255': '056.tiff', '40282270': '051.tiff', '40282261': '054.tiff', '40282258': '060.tiff', '40282264': '058.tiff', '40282267': '062.tiff', '40282276': '057.tiff', '40282273': '064.tiff', '40282279': '059.tiff', '40282282': '066.tiff', '40282288': '068.tiff', '40282285': '061.tiff', '40282291': '063.tiff', '40282294': '065.tiff', '40282297': '067.tiff', '40282300': '070.tiff', '40282303': '076.tiff', '40282309': '069.tiff', '40282306': '074.tiff', '40282318': '072.tiff', '40282312': '078.tiff', '40282321': '073.tiff', '40282315': '080.tiff', '40282324': '071.tiff', '40282327': '082.tiff', '40282330': '075.tiff', '40282333': '077.tiff', '40282336': '086.tiff', '40282339': '081.tiff', '40282342': '083.tiff', '40282351': '079.tiff', '40282357': '084.tiff', '40282345': '096.tiff', '40282348': '092.tiff', '40282354': '085.tiff', '40282363': '090.tiff', '40282366': '088.tiff', '40282360': '094.tiff', '40282369': '098.tiff', '40282372': '087.tiff', '40282378': '089.tiff', '40282375': '091.tiff', '40282381': '093.tiff', '40282384': '100.tiff', '40282387': '095.tiff', '40282390': '102.tiff', '40282393': '097.tiff', '40282396': '104.tiff', '40282405': '099.tiff', '40282408': '116.tiff', '40282402': '108.tiff', '40282414': '101.tiff', '40282417': '106.tiff', '40282429': '112.tiff', '40282420': '110.tiff', '40282411': '103.tiff', '40282426': '127.tiff', '40282423': '105.tiff', '40282432': '107.tiff', '40282435': '109.tiff', '40282438': '118.tiff', '40282441': '114.tiff', '40282444': '111.tiff', '40282447': '124.tiff', '40282450': '113.tiff', '40282453': '122.tiff', '40282459': '120.tiff', '40282456': '115.tiff', '40282462': '117.tiff', '40282465': '128.tiff', '40282468': '119.tiff', '40282471': '126.tiff', '40282474': '125.tiff', '40282477': '123.tiff', '40282480': '131.tiff', '40282483': '129.tiff', '40282486': '121.tiff', '40282492': '133.tiff', '40282489': '135.tiff', '40282495': '137.tiff', '40282501': '139.tiff', '40282498': '134.tiff', '40282504': '136.tiff', '40282507': '132.tiff', '40282516': '130.tiff', '40282510': '138.tiff', '40282513': '140.tiff', '40282519': '143.tiff', '40282522': '142.tiff', '40282528': '145.tiff', '40282531': '148.tiff', '40282525': '146.tiff', '40282540': '144.tiff', '40282534': '141.tiff', '40282537': '150.tiff', '40282543': '202.tiff', '40282546': '204.tiff', '40282552': '205.tiff', '40282549': '203.tiff', '40282555': '147.tiff', '40282558': '201.tiff', '40282564': '149.tiff', '40282561': '207.tiff', '40282570': '206.tiff', '40282573': '209.tiff', '40282567': '208.tiff', '40282576': '213.tiff', '40282594': '210.tiff', '40282603': '217.tiff', '40282600': '215.tiff', '40282606': '222.tiff', '40282609': '220.tiff', '40282612': '211.tiff', '40282618': '214.tiff', '40282615': '216.tiff', '40282621': '212.tiff', '40282624': '224.tiff', '40282627': '218.tiff', '40282630': '223.tiff', '40282639': '221.tiff', '40282633': '229.tiff', '40282636': '219.tiff', '40282642': '227.tiff', '40282648': '231.tiff', '40282645': '235.tiff', '40282651': '233.tiff', '40282654': '237.tiff', '40282660': '239.tiff', '40282657': '225.tiff', '40282666': '226.tiff', '40282663': '241.tiff', '40282669': '230.tiff', '40282672': '228.tiff', '40282675': '232.tiff', '40282678': '243.tiff', '40282681': '234.tiff', '40282684': '236.tiff', '40282687': '238.tiff', '40282690': '249.tiff', '40282693': '247.tiff', '40282696': '250.tiff', '40282699': '252.tiff', '40282705': '254.tiff', '40282702': '256.tiff', '40282708': '240.tiff', '40282711': '258.tiff', '40282714': '242.tiff', '40282717': '244.tiff', '40282720': '245.tiff', '40282723': '246.tiff', '40282726': '262.tiff', '40282729': '264.tiff', '40282732': '251.tiff', '40282741': '266.tiff', '40282738': '270.tiff', '40282735': '268.tiff', '40282759': '272.tiff', '40282744': '276.tiff', '40282747': '274.tiff', '40282753': '248.tiff', '40282756': '253.tiff', '40282762': '255.tiff', '40282768': '259.tiff', '40282765': '257.tiff', '40282771': '260.tiff', '40282774': '261.tiff', '40282777': '278.tiff', '40282780': '280.tiff', '40282783': '288.tiff', '40282789': '286.tiff', '40282786': '284.tiff', '40282792': '290.tiff', '40282813': '263.tiff', '40282795': '265.tiff', '40282798': '267.tiff', '40282801': '282.tiff', '40282804': '292.tiff', '40282807': '269.tiff', '40282810': '273.tiff', '40282819': '271.tiff', '40282816': '294.tiff', '40282822': '275.tiff', '40282828': '277.tiff', '40282831': '296.tiff', '40282825': '298.tiff', '40282834': '300.tiff', '40282837': '304.tiff', '40282843': '302.tiff', '40282840': '281.tiff', '40282846': '306.tiff', '40282849': '279.tiff', '40282852': '285.tiff', '40282858': '283.tiff', '40282861': '308.tiff', '40282855': '289.tiff', '40282864': '287.tiff', '40282867': '291.tiff', '40282870': '314.tiff', '40282876': '293.tiff', '40282873': '312.tiff', '40282879': '310.tiff', '40282882': '316.tiff', '40282888': '295.tiff', '40282885': '317.tiff', '40282891': '319.tiff', '40282900': '299.tiff', '40282894': '297.tiff', '40282897': '303.tiff', '40282903': '301.tiff', '40282906': '322.tiff', '40282909': '326.tiff', '40282921': '324.tiff', '40282912': '327.tiff', '40282915': '305.tiff', '40282918': '307.tiff', '40282924': '330.tiff', '40282927': '332.tiff', '40282930': '335.tiff', '40282933': '309.tiff', '40282936': '337.tiff', '40282939': '311.tiff', '40282942': '339.tiff', '40282945': '315.tiff', '40282948': '318.tiff', '40282951': '313.tiff', '40282954': '342.tiff', '40282957': '320.tiff', '40282960': '341.tiff', '40282969': '323.tiff', '40282963': '321.tiff', '40282966': '344.tiff', '40282972': '346.tiff', '40282975': '348.tiff', '40282978': '325.tiff', '40282984': '350.tiff', '40282987': '352.tiff', '40282981': '354.tiff', '40282990': '329.tiff', '40282993': '328.tiff', '40283002': '356.tiff', '40282996': '333.tiff', '40282999': '331.tiff', '40283005': '334.tiff', '40283008': '338.tiff', '40283014': '336.tiff', '40283017': '357.tiff', '40283020': '361.tiff', '40283023': '359.tiff', '40283032': '363.tiff', '40283029': '340.tiff', '40283026': '365.tiff', '40283044': '347.tiff', '40283038': '343.tiff', '40283035': '372.tiff', '40283041': '351.tiff', '40283047': '349.tiff', '40283050': '374.tiff', '40283053': '376.tiff', '40283059': '367.tiff', '40283056': '353.tiff', '40283062': '355.tiff', '40283065': '380.tiff', '40283068': '378.tiff', '40283071': '345.tiff', '40283074': '358.tiff', '40283077': '385.tiff', '40283080': '389.tiff', '40283089': '387.tiff', '40283083': '362.tiff', '40283086': '360.tiff', '40283092': '366.tiff', '40283104': '397.tiff', '40283098': '391.tiff', '40283107': '393.tiff', '40283095': '368.tiff', '40283101': '364.tiff', '40283110': '395.tiff', '40283119': '369.tiff', '40283113': '371.tiff', '40283116': '370.tiff', '40283122': '399.tiff', '40283125': '402.tiff', '40283128': '404.tiff', '40283131': '373.tiff', '40283134': '375.tiff', '40283137': '377.tiff', '40283140': '410.tiff', '40283143': '408.tiff', '40283155': '413.tiff', '40283152': '415.tiff', '40283146': '379.tiff', '40283149': '381.tiff', '40283158': '382.tiff', '40283161': '417.tiff', '40283164': '406.tiff', '40283167': '383.tiff', '40283170': '419.tiff', '40283176': '421.tiff', '40283173': '388.tiff', '40283179': '384.tiff', '40283182': '390.tiff', '40283188': '386.tiff', '40283194': '423.tiff', '40283200': '392.tiff', '40283221': '429.tiff', '40283206': '431.tiff', '40283242': '427.tiff', '40283212': '425.tiff', '40283236': '394.tiff', '40283224': '432.tiff', '40283245': '396.tiff', '40283257': '433.tiff', '40283263': '435.tiff', '40283278': '398.tiff', '40283287': '400.tiff', '40283293': '401.tiff', '40283302': '403.tiff', '40283311': '405.tiff', '40283320': '436.tiff', '40283326': '407.tiff', '40283329': '438.tiff', '40283332': '440.tiff', '40283335': '409.tiff', '40283338': '411.tiff', '40283344': '442.tiff', '40283341': '444.tiff', '40283350': '412.tiff', '40283347': '446.tiff', '40283353': '450.tiff', '40283356': '448.tiff', '40283365': '414.tiff', '40283362': '416.tiff', '40283359': '452.tiff', '40283368': '418.tiff', '40283371': '420.tiff', '40283374': '454.tiff', '40283377': '456.tiff', '40283386': '422.tiff', '40283380': '426.tiff', '40283383': '457.tiff', '40283389': '424.tiff', '40283392': '428.tiff', '40283395': '461.tiff', '40283401': '459.tiff', '40283407': '430.tiff', '40283410': '464.tiff', '40283413': '466.tiff', '40283416': '468.tiff', '40283419': '434.tiff', '40283425': '439.tiff', '40283422': '470.tiff', '40283428': '437.tiff', '40283434': '441.tiff', '40283437': '472.tiff', '40283431': '445.tiff', '40283443': '474.tiff', '40283449': '476.tiff', '40283440': '443.tiff', '40283446': '447.tiff', '40283452': '449.tiff', '40283458': '478.tiff', '40283455': '482.tiff', '40283461': '451.tiff', '40283464': '480.tiff', '40283473': '455.tiff', '40283467': '486.tiff', '40283476': '484.tiff', '40283479': '453.tiff', '40283491': '492.tiff', '40283482': '491.tiff', '40283488': '458.tiff', '40283494': '489.tiff', '40283497': '460.tiff', '40283503': '463.tiff', '40283500': '494.tiff', '40283509': '462.tiff', '40283506': '465.tiff', '40283512': '497.tiff', '40283515': '501.tiff', '40283521': '467.tiff', '40283518': '499.tiff', '40283524': '469.tiff', '40283527': '471.tiff', '40283533': '503.tiff', '40283530': '505.tiff', '40283539': '473.tiff', '40283536': '475.tiff', '40283548': '507.tiff', '40283545': '477.tiff', '40283542': '509.tiff', '40283554': '511.tiff', '40283557': '514.tiff', '40283563': '479.tiff', '40283560': '481.tiff', '40283566': '515.tiff', '40283569': '517.tiff', '40283572': '483.tiff', '40283575': '487.tiff', '40283584': '520.tiff', '40283587': '488.tiff', '40283593': '485.tiff', '40283581': '522.tiff', '40283578': '524.tiff', '40283596': '490.tiff', '40283599': '493.tiff', '40283602': '526.tiff', '40283611': '528.tiff', '40283608': '530.tiff', '40283605': '495.tiff', '40283614': '496.tiff', '40283617': '498.tiff', '40283620': '532.tiff', '40283623': '535.tiff', '40283632': '500.tiff', '40283629': '538.tiff', '40283635': '540.tiff', '40283638': '542.tiff', '40283641': '502.tiff', '40283644': '504.tiff', '40283647': '506.tiff', '40283653': '544.tiff', '40283650': '508.tiff', '40283659': '548.tiff', '40283656': '510.tiff', '40283665': '546.tiff', '40283662': '550.tiff', '40283668': '512.tiff', '40283677': '513.tiff', '40283671': '552.tiff', '40283674': '516.tiff', '40283680': '518.tiff', '40283683': '521.tiff', '40283689': '519.tiff', '40283686': '523.tiff', '40283692': '525.tiff', '40283695': '527.tiff', '40283698': '531.tiff', '40283701': '529.tiff', '40283704': '533.tiff', '40283707': '534.tiff', '40283710': '537.tiff', '40283716': '541.tiff', '40283719': '536.tiff', '40283713': '539.tiff', '40283722': '543.tiff', '40283725': '545.tiff', '40283728': '547.tiff', '40283731': '549.tiff', '40283734': '551.tiff', '40283740': '553.tiff', '41265615': 'MIDOGpp.json'}


    _DOMAIN_SPLITS = [
        ("1a", 0, 50, ""),
        ("1b", 50, 100, "csid"),
        ("1c", 100, 150, "csid"),
        ("2", 200, 244, "near"),
        ("3", 244, 299, "near"),
        ("4", 299, 349, "near"),
        ("5", 349, 404, "near"),
        ("6a", 404, 489, "near"),
        ("6b", 489, 504, "near"),
        ("7", 504, 553, "near"),
    ]

    def download(self) -> None:
        tmp_download = join(self.root, "_tmp_midog")

        # Check if any slide is missing across any domain
        json_path = join(self.root, "MIDOGpp.json")
        if not exists(json_path):
            json_tmp = join(tmp_download, "MIDOGpp.json")
            if not exists(json_tmp):
                os.makedirs(tmp_download, exist_ok=True)
                _download_url_with_ssl_fallback("https://ndownloader.figshare.com/files/41265615", tmp_download, filename="MIDOGpp.json")
            json_path = json_tmp

        with open(json_path) as f:
            midog_data = json.load(f)

        fname_to_fid = {fname: fid for fid, fname in self._FIGSHARE_FILES.items()}

        slides_to_download = []
        for domain_name, start, end, subfolder in self._DOMAIN_SPLITS:
            out_dir = join(self.root, subfolder, domain_name) if subfolder else join(self.root, domain_name)
            for img_index in range(start, min(end, len(midog_data["images"]))):
                img_data = midog_data["images"][img_index]
                img_id = img_data["id"]
                img_dir = join(out_dir, f"{img_id:03d}")
                if not isdir(img_dir) or len(os.listdir(img_dir)) == 0:
                    slides_to_download.append((domain_name, out_dir, img_data))

        if not slides_to_download:
            return

        log.info(
            f"MIDOG has {len(slides_to_download)} missing slide(s) to process. "
            "Downloading from Figshare and extracting 50x50 TIFF patches..."
        )
        os.makedirs(tmp_download, exist_ok=True)

        for domain_name, out_dir, img_data in slides_to_download:
            img_id = img_data["id"]
            img_name = img_data["file_name"]
            img_dir = join(out_dir, f"{img_id:03d}")
            os.makedirs(img_dir, exist_ok=True)

            img_path = join(tmp_download, img_name)
            if not exists(img_path):
                fid = fname_to_fid.get(img_name)
                if fid:
                    url = f"https://ndownloader.figshare.com/files/{fid}"
                    log.info(f"Downloading {img_name} ({fid}) for domain {domain_name}...")
                    _download_url_with_ssl_fallback(url, tmp_download, filename=img_name)

            if not exists(img_path):
                continue

            image = Image.open(img_path)
            img_dims = (img_data["width"], img_data["height"])
            annotations = [ann for ann in midog_data["annotations"] if ann["image_id"] == img_id]

            for ann in annotations:
                label = ann["category_id"]
                bbox = ann["bbox"]
                ann_id = ann["id"]

                if bbox[2] - bbox[0] == 50 and bbox[3] - bbox[1] == 50:
                    crop = image.crop(bbox)
                    crop.save(join(img_dir, f"{img_id:03d}_{ann_id}_{label}.tiff"))

                    add_bbox = [bbox[0] + 100, bbox[1], bbox[2] + 100, bbox[3]]
                    if _validate_patch(img_dims, annotations, add_bbox):
                        crop_neg = image.crop(add_bbox)
                        crop_neg.save(join(img_dir, f"{img_id:03d}_{ann_id}_0.tiff"))

            if exists(img_path):
                os.remove(img_path)

        shutil.rmtree(tmp_download, ignore_errors=True)


class CCAgT(OpenMIBOODDataset):
    """
    Cervical cell cytology dataset (Far-OOD for MIDOG).
    Downloads archive from Mendeley Data and crops 50x50 nuclei patches.

    :see Mendeley: `CCAgT Dataset <https://data.mendeley.com/datasets/wg4bpm33hj/2>`__
    """

    url = "https://data.mendeley.com/public-api/zip/wg4bpm33hj/download/2"
    filename = "ccagt.zip"
    target_rel_dir = "far/ccagt_crops"

    def download(self) -> None:
        dest_dir = join(self.root, self.target_rel_dir)
        if isdir(dest_dir) and len(os.listdir(dest_dir)) > 0:
            return

        archive_path = join(self.root, self.filename)
        if not exists(archive_path):
            archive_path = _download_url_with_ssl_fallback(
                self.url, self.root, filename=self.filename, md5=self.tgz_md5
            )

        log.info(f"Extracting CCAgT zip -> {dest_dir}")
        tmp_output = join(self.root, "_tmp_ccagt")
        os.makedirs(tmp_output, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmp_output)

        subsets_dir = join(tmp_output, "subsets")
        os.makedirs(subsets_dir, exist_ok=True)
        images_dir = join(tmp_output, "wg4bpm33hj-2", "images")
        if isdir(images_dir):
            for zip_name in os.listdir(images_dir):
                if zip_name.endswith(".zip"):
                    sub_zip = join(images_dir, zip_name)
                    with zipfile.ZipFile(sub_zip, "r") as szf:
                        szf.extractall(subsets_dir)

        coco_json = join(tmp_output, "wg4bpm33hj-2", "CCAgT_COCO_OD.json")
        if exists(coco_json):
            with open(coco_json) as f:
                ccagt_data = json.load(f)

            os.makedirs(dest_dir, exist_ok=True)
            for ann in ccagt_data["annotations"]:
                cat_id = ann["category_id"]
                if cat_id in (2, 3):
                    continue
                bbox = [int(c) for c in ann["bbox"]]
                cx, cy = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
                crop_box = [cx - 25, cy - 25, cx + 25, cy + 25]

                image_id = ann["image_id"]
                img_entry = [img for img in ccagt_data["images"] if img["id"] == image_id]
                if not img_entry:
                    continue
                file_name = img_entry[0]["file_name"]
                slide_id = file_name.split("_")[0]
                img_src = join(subsets_dir, slide_id, f"{file_name}.jpg")
                if exists(img_src):
                    out_slide_dir = join(dest_dir, file_name)
                    os.makedirs(out_slide_dir, exist_ok=True)
                    img = Image.open(img_src)
                    if 0 <= crop_box[0] and crop_box[2] < img.width and 0 <= crop_box[1] and crop_box[3] < img.height:
                        crop = img.crop(crop_box)
                        crop.save(join(out_slide_dir, f"{ann['id']}_{cat_id}.jpg"))

        shutil.rmtree(tmp_output, ignore_errors=True)


class FNAC2019(OpenMIBOODDataset):
    """
    Fine-needle aspirate cytology (FNAC 2019) dataset (Far-OOD for MIDOG).
    
    .. warning::
        Automatic download of FNAC 2019 requires manual setup due to dynamic token authentication.
    """

    target_rel_dir = "far/fnac2019_crops"

    def download(self) -> None:
        if isdir(self.basedir) and len(os.listdir(self.basedir)) > 0:
            return

        raise RuntimeError(
            "Automatic download of FNAC 2019 requires manual setup due to dynamic token authentication.\n"
            "Please run OpenMIBOOD's download_midog_fnac2019.py script to extract patches to far/fnac2019_crops/."
        )


# =====================================================================
# OASIS-3 Benchmark Datasets
# =====================================================================


class CHAOS(OpenMIBOODDataset):
    """
    Combined Healthy Abdominal Organ Segmentation (CHAOS) MRI dataset (Far-OOD for OASIS-3).
    Converts DICOM slices to NIfTI volumes.

    .. note::
        Automatic conversion from DICOM to NIfTI requires the ``SimpleITK`` package. 
        Please install it manually (``pip install SimpleITK``). If it is not installed, 
        the raw DICOM folders will be copied without conversion.

    :see Zenodo: `CHAOS Test Sets <https://zenodo.org/records/3431873>`__
    """

    url = "https://zenodo.org/records/3431873/files/CHAOS_Test_Sets.zip?download=1"
    filename = "CHAOS_Test_Sets.zip"
    target_rel_dir = "far/CHAOS/NIFTI/InPhase"

    def download(self) -> None:
        dest_dir = join(self.root, self.target_rel_dir)
        if isdir(dest_dir) and len(os.listdir(dest_dir)) > 0:
            return

        archive_path = _download_url_with_ssl_fallback(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

        tmp_extract = join(self.root, "_tmp_chaos")
        os.makedirs(tmp_extract, exist_ok=True)
        log.info(f"Extracting CHAOS -> {tmp_extract}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmp_extract)

        os.makedirs(dest_dir, exist_ok=True)

        try:
            import SimpleITK as sitk

            mr_path = join(tmp_extract, "Test_Sets", "MR")
            if not exists(mr_path):
                mr_path = join(tmp_extract, "MR")

            if exists(mr_path):
                for subject in os.listdir(mr_path):
                    in_phase_path = join(mr_path, subject, "T1DUAL", "DICOM_anon", "InPhase")
                    if exists(in_phase_path):
                        reader = sitk.ImageSeriesReader()
                        dicom_names = reader.GetGDCMSeriesFileNames(in_phase_path)
                        reader.SetFileNames(dicom_names)
                        image = reader.Execute()
                        nii_out = join(dest_dir, f"{subject}.nii.gz")
                        sitk.WriteImage(image, nii_out)
        except ImportError:
            log.warning("SimpleITK not installed; copying raw DICOM folders instead.")
            raw_target = join(self.root, "far/CHAOS/raw")
            os.makedirs(raw_target, exist_ok=True)
            shutil.copytree(tmp_extract, raw_target, dirs_exist_ok=True)

        shutil.rmtree(tmp_extract, ignore_errors=True)


class Task02Heart(OpenMIBOODDataset):
    """
    MSD Task02 Heart MRI dataset (Far-OOD for OASIS-3).

    :see Medical Segmentation Decathlon: `MSD <http://medicaldecathlon.com/>`__
    """

    gdrive_id = "1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY"
    filename = "Task02_Heart.tar"
    target_rel_dir = "far/SegmentationDecathlon/Task02_Heart"

    def download(self) -> None:
        dest_dir = join(self.root, self.target_rel_dir)
        if isdir(dest_dir) and len(os.listdir(dest_dir)) > 0:
            return

        archive_path = join(self.root, self.filename)
        if not exists(archive_path):
            try:
                import gdown

                gdown.download(id=self.gdrive_id, output=archive_path, quiet=False)
            except ImportError as e:
                raise RuntimeError(
                    "Downloading Task02 Heart requires 'gdown' (pip install gdown)."
                ) from e

        log.info(f"Extracting Task02 Heart -> {dest_dir}")
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        with tarfile.open(archive_path, "r") as tf:
            tf.extractall(os.path.dirname(dest_dir))


class OASIS3(OpenMIBOODDataset):
    """
    In-distribution brain MRI dataset from OASIS-3 (3 classes: CN, MCI, AD).

    .. warning::
        Automatic download is not possible because data access requires registration.
        Request access at https://sites.wustl.edu/oasisbrains/home/access/ and download
        via your NITRC account.
    """

    target_rel_dir = "OASIS3"

    def download(self) -> None:
        if isdir(self.basedir) and len(os.listdir(self.basedir)) > 0:
            return

        raise RuntimeError(
            "Automatic download of OASIS-3 is not possible because data access requires registration.\n"
            "1. Request access at https://sites.wustl.edu/oasisbrains/home/access/\n"
            "2. Download via your NITRC account and follow OpenMIBOOD setup instructions."
        )


class BraTS(OpenMIBOODDataset):
    """
    BraTS 2023 glioma MRI dataset (Near-OOD for OASIS-3).
    Requires Synapse challenge registration.
    """

    target_rel_dir = "near/BraTS2023-GLI"

    def download(self) -> None:
        if isdir(self.basedir) and len(os.listdir(self.basedir)) > 0:
            return

        raise RuntimeError(
            "Automatic download of BraTS 2023 is not possible.\n"
            "Please register at https://www.synapse.org/Synapse:syn51156910/wiki/627000 "
            "and follow OpenMIBOOD instructions."
        )


class ATLAS(OpenMIBOODDataset):
    """
    ATLAS R2.0 stroke lesion MRI dataset (Near-OOD for OASIS-3).

    .. warning::
        Automatic download is not possible.
        Please agree to terms at https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html
        and submit the form to acquire the decryption key.
    """

    target_rel_dir = "near/ATLAS"

    def download(self) -> None:
        if isdir(self.basedir) and len(os.listdir(self.basedir)) > 0:
            return

        raise RuntimeError(
            "Automatic download of ATLAS R2.0 is not possible.\n"
            "Please agree to terms at https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html "
            "and submit the form to acquire the decryption key."
        )


# =====================================================================
# Unified download helper
# =====================================================================


def download_all_openmibood(root: str, benchmark: str = "all") -> None:
    """
    Download and extract all publicly accessible datasets for OpenMIBOOD into `root`.

    :param root: destination root directory matching the layout expected by OpenMIBOOD benchmarks
    :param benchmark: one of ``'phakir'``, ``'midog'``, ``'oasis3'``, or ``'all'``
    """
    benchmarks = ["phakir", "midog", "oasis3"] if benchmark == "all" else [benchmark.lower()]

    for b in benchmarks:
        if b == "phakir":
            for cls in (Cholec80, KvasirSEG, CATARACTS):
                try:
                    cls(root=root, download=True)
                except Exception as e:
                    log.error(f"Failed downloading {cls.__name__}: {e}")

        elif b == "oasis3":
            for cls in (CHAOS, Task02Heart):
                try:
                    cls(root=root, download=True)
                except Exception as e:
                    log.error(f"Failed downloading {cls.__name__}: {e}")

        elif b == "midog":
            for cls in (CCAgT,):
                try:
                    cls(root=root, download=True)
                except Exception as e:
                    log.error(f"Failed downloading {cls.__name__}: {e}")
        else:
            raise ValueError(f"Unknown benchmark: {b}. Must be one of 'phakir', 'midog', 'oasis3', 'all'")
