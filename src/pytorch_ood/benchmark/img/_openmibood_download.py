"""
Download and preparation of the datasets used by the OpenMIBOOD benchmarks.

The functions port the download scripts of the
`OpenMIBOOD repository <https://github.com/remic-othr/OpenMIBOOD/tree/main/scripts/download/OpenMIBOOD>`__
and write the data to the paths referenced by the OpenMIBOOD imglists. Datasets that require
registration have no download function and must be prepared manually, as described in the
documentation of the benchmarks.
"""

import glob
import json
import logging
import os
import shutil
import ssl
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from os.path import exists, isdir, join
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from torchvision.datasets.utils import check_integrity, download_url

log = logging.getLogger(__name__)


def _is_nonempty_dir(path: str) -> bool:
    return isdir(path) and len(os.listdir(path)) > 0


@dataclass(frozen=True)
class _Source:
    """
    A dataset used by an OpenMIBOOD benchmark.
    """

    #: name of the dataset, used in messages
    name: str
    #: directories below the benchmark root that hold the prepared data
    targets: Tuple[str, ...]
    #: function that downloads and prepares the data below the given root, ``None`` if the
    #: dataset has to be prepared manually
    download: Optional[Callable[[str], None]] = None

    def is_present(self, root: str) -> bool:
        return all(_is_nonempty_dir(join(root, t)) for t in self.targets)


def _is_certificate_error(e: Optional[BaseException]) -> bool:
    """
    Whether the exception, or an exception it was raised from, is an SSL certificate error.
    """
    while e is not None:
        if isinstance(e, ssl.SSLCertVerificationError):
            return True
        if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            return True
        e = e.__cause__ or e.__context__
    return False


def _download(url: str, root: str, filename: str, md5: Optional[str] = None) -> str:
    """
    Download a file to ``root/filename`` and return its path.

    The file is first written to ``<filename>.part`` and only renamed after the download
    completed (and its MD5 checksum matched, if given), so an interrupted download is never
    mistaken for a complete file. Some servers (e.g. Simula) omit intermediate certificates in
    their TLS handshake. If the certificate can not be verified and an MD5 checksum is given, the
    download is retried without certificate verification, since the checksum still verifies the
    file.
    """
    os.makedirs(root, exist_ok=True)
    fpath = join(root, filename)
    if exists(fpath) and (md5 is None or check_integrity(fpath, md5)):
        return fpath

    part = filename + ".part"
    ppath = join(root, part)
    if exists(ppath):
        os.remove(ppath)

    try:
        download_url(url, root, filename=part, md5=md5)
    except Exception as e:
        if md5 is None or not _is_certificate_error(e):
            raise
        log.warning(
            f"Could not verify the certificate of {url} ({e}). Retrying without certificate "
            "verification, the file will be verified with its MD5 checksum."
        )
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as response:
            with open(ppath, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

    if md5 is not None and not check_integrity(ppath, md5):
        os.remove(ppath)
        raise RuntimeError(f"Download of {url} failed the MD5 integrity check.")

    os.replace(ppath, fpath)
    return fpath


def _extract_zip(archive: str, to: str) -> None:
    log.info(f"Extracting {archive} -> {to}")
    os.makedirs(to, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(to)


def _extract_tar(archive: str, to: str) -> None:
    log.info(f"Extracting {archive} -> {to}")
    os.makedirs(to, exist_ok=True)
    with tarfile.open(archive, "r") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(to, filter="data")
        else:
            # Python versions without extraction filters: reject members outside of the target
            base = os.path.realpath(to)
            for member in tf.getmembers():
                path = os.path.realpath(join(to, member.name))
                if os.path.commonpath([base, path]) != base or member.issym() or member.islnk():
                    raise RuntimeError(f"Refusing to extract unsafe archive member {member.name}")
            tf.extractall(to)


def _download_and_move(url: str, md5: Optional[str], root: str, src: str, target: str) -> None:
    """
    Download a zip archive, extract it to a temporary directory below ``root`` and move its
    subdirectory ``src`` to ``root/target``. The archive is deleted afterwards.
    """
    filename = url.split("/")[-1].split("?")[0]
    archive = _download(url, root, filename, md5=md5)
    tmp = join(root, f"_tmp_{filename}")
    shutil.rmtree(tmp, ignore_errors=True)
    _extract_zip(archive, tmp)
    dest = join(root, target)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(join(tmp, src), dest)
    shutil.rmtree(tmp, ignore_errors=True)
    os.remove(archive)


def _overlaps_none(img_dims, annotations: List[Dict[str, Any]], bbox: List[int]) -> bool:
    """
    Whether ``bbox`` lies inside the image and none of its corners lies inside an annotation
    (``validate_patch`` of the OpenMIBOOD scripts).
    """
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= img_dims[0] or bbox[3] >= img_dims[1]:
        return False

    for annotation in annotations:
        a = annotation["bbox"]
        if a[0] < bbox[0] < a[2] and a[1] < bbox[1] < a[3]:
            return False
        if a[0] < bbox[2] < a[2] and a[1] < bbox[3] < a[3]:
            return False

    return True


# =====================================================================
# PhaKIR benchmark
# =====================================================================


def _download_kvasir(root: str) -> None:
    _download_and_move(
        "https://datasets.simula.no/downloads/kvasir-seg.zip",
        "6323d9094df93b35d43069a566ee1ca3",
        root,
        src="Kvasir-SEG/images",
        target="far/kvasir-seg",
    )


def _download_cataracts(root: str) -> None:
    # cleaned subset prepared by the OpenMIBOOD authors
    _download_and_move(
        "https://zenodo.org/records/14924735/files/CATARACTS.zip?download=1",
        "fa5cc2974c5edc93ece42da634a1d598",
        root,
        src="CATARACTS",
        target="far/CATARACTS",
    )


def _download_cholec80(root: str) -> None:
    # frames cropped by the OpenMIBOOD authors to remove the black borders
    _download_and_move(
        "https://zenodo.org/records/14921670/files/Cholec80_cropped.zip?download=1",
        "716b52b1d103f47b72826e83b4da1e04",
        root,
        src="Cholec80_cropped",
        target="near/Cholec80_cropped",
    )


PHAKIR = _Source("PhaKIR", ("Video_02",))
CHOLEC80 = _Source("Cholec80", ("near/Cholec80_cropped",), _download_cholec80)
ENDOVIS2015 = _Source("EndoVis 2015", ("near/Endovis2015_cropped",))
ENDOVIS2018 = _Source("EndoVis 2018", ("near/Endovis2018",))
KVASIR = _Source("Kvasir-SEG", ("far/kvasir-seg",), _download_kvasir)
CATARACTS = _Source("CATARACTS", ("far/CATARACTS",), _download_cataracts)


# =====================================================================
# MIDOG benchmark
# =====================================================================

#: Figshare file IDs of the MIDOG++ slides
_MIDOG_FIGSHARE_FILES = {
    "40282102": "002.tiff",
    "40282099": "001.tiff",
    "40282096": "016.tiff",
    "40282105": "008.tiff",
    "40282111": "003.tiff",
    "40282108": "010.tiff",
    "40282114": "014.tiff",
    "40282132": "012.tiff",
    "40282129": "011.tiff",
    "40282126": "006.tiff",
    "40282117": "013.tiff",
    "40282123": "015.tiff",
    "40282135": "007.tiff",
    "40282120": "005.tiff",
    "40282138": "004.tiff",
    "40282141": "009.tiff",
    "40282144": "018.tiff",
    "40282147": "022.tiff",
    "40282153": "017.tiff",
    "40282150": "020.tiff",
    "40282162": "021.tiff",
    "40282156": "026.tiff",
    "40282159": "023.tiff",
    "40282165": "028.tiff",
    "40282180": "019.tiff",
    "40282171": "024.tiff",
    "40282186": "034.tiff",
    "40282183": "030.tiff",
    "40282168": "031.tiff",
    "40282174": "027.tiff",
    "40282177": "025.tiff",
    "40282189": "029.tiff",
    "40282192": "038.tiff",
    "40282195": "036.tiff",
    "40282198": "032.tiff",
    "40282204": "037.tiff",
    "40282207": "033.tiff",
    "40282201": "035.tiff",
    "40282210": "039.tiff",
    "40282213": "042.tiff",
    "40282225": "040.tiff",
    "40282216": "044.tiff",
    "40282219": "048.tiff",
    "40282222": "046.tiff",
    "40282234": "043.tiff",
    "40282231": "041.tiff",
    "40282228": "047.tiff",
    "40282237": "052.tiff",
    "40282240": "050.tiff",
    "40282243": "045.tiff",
    "40282246": "049.tiff",
    "40282252": "053.tiff",
    "40282249": "055.tiff",
    "40282255": "056.tiff",
    "40282270": "051.tiff",
    "40282261": "054.tiff",
    "40282258": "060.tiff",
    "40282264": "058.tiff",
    "40282267": "062.tiff",
    "40282276": "057.tiff",
    "40282273": "064.tiff",
    "40282279": "059.tiff",
    "40282282": "066.tiff",
    "40282288": "068.tiff",
    "40282285": "061.tiff",
    "40282291": "063.tiff",
    "40282294": "065.tiff",
    "40282297": "067.tiff",
    "40282300": "070.tiff",
    "40282303": "076.tiff",
    "40282309": "069.tiff",
    "40282306": "074.tiff",
    "40282318": "072.tiff",
    "40282312": "078.tiff",
    "40282321": "073.tiff",
    "40282315": "080.tiff",
    "40282324": "071.tiff",
    "40282327": "082.tiff",
    "40282330": "075.tiff",
    "40282333": "077.tiff",
    "40282336": "086.tiff",
    "40282339": "081.tiff",
    "40282342": "083.tiff",
    "40282351": "079.tiff",
    "40282357": "084.tiff",
    "40282345": "096.tiff",
    "40282348": "092.tiff",
    "40282354": "085.tiff",
    "40282363": "090.tiff",
    "40282366": "088.tiff",
    "40282360": "094.tiff",
    "40282369": "098.tiff",
    "40282372": "087.tiff",
    "40282378": "089.tiff",
    "40282375": "091.tiff",
    "40282381": "093.tiff",
    "40282384": "100.tiff",
    "40282387": "095.tiff",
    "40282390": "102.tiff",
    "40282393": "097.tiff",
    "40282396": "104.tiff",
    "40282405": "099.tiff",
    "40282408": "116.tiff",
    "40282402": "108.tiff",
    "40282414": "101.tiff",
    "40282417": "106.tiff",
    "40282429": "112.tiff",
    "40282420": "110.tiff",
    "40282411": "103.tiff",
    "40282426": "127.tiff",
    "40282423": "105.tiff",
    "40282432": "107.tiff",
    "40282435": "109.tiff",
    "40282438": "118.tiff",
    "40282441": "114.tiff",
    "40282444": "111.tiff",
    "40282447": "124.tiff",
    "40282450": "113.tiff",
    "40282453": "122.tiff",
    "40282459": "120.tiff",
    "40282456": "115.tiff",
    "40282462": "117.tiff",
    "40282465": "128.tiff",
    "40282468": "119.tiff",
    "40282471": "126.tiff",
    "40282474": "125.tiff",
    "40282477": "123.tiff",
    "40282480": "131.tiff",
    "40282483": "129.tiff",
    "40282486": "121.tiff",
    "40282492": "133.tiff",
    "40282489": "135.tiff",
    "40282495": "137.tiff",
    "40282501": "139.tiff",
    "40282498": "134.tiff",
    "40282504": "136.tiff",
    "40282507": "132.tiff",
    "40282516": "130.tiff",
    "40282510": "138.tiff",
    "40282513": "140.tiff",
    "40282519": "143.tiff",
    "40282522": "142.tiff",
    "40282528": "145.tiff",
    "40282531": "148.tiff",
    "40282525": "146.tiff",
    "40282540": "144.tiff",
    "40282534": "141.tiff",
    "40282537": "150.tiff",
    "40282543": "202.tiff",
    "40282546": "204.tiff",
    "40282552": "205.tiff",
    "40282549": "203.tiff",
    "40282555": "147.tiff",
    "40282558": "201.tiff",
    "40282564": "149.tiff",
    "40282561": "207.tiff",
    "40282570": "206.tiff",
    "40282573": "209.tiff",
    "40282567": "208.tiff",
    "40282576": "213.tiff",
    "40282594": "210.tiff",
    "40282603": "217.tiff",
    "40282600": "215.tiff",
    "40282606": "222.tiff",
    "40282609": "220.tiff",
    "40282612": "211.tiff",
    "40282618": "214.tiff",
    "40282615": "216.tiff",
    "40282621": "212.tiff",
    "40282624": "224.tiff",
    "40282627": "218.tiff",
    "40282630": "223.tiff",
    "40282639": "221.tiff",
    "40282633": "229.tiff",
    "40282636": "219.tiff",
    "40282642": "227.tiff",
    "40282648": "231.tiff",
    "40282645": "235.tiff",
    "40282651": "233.tiff",
    "40282654": "237.tiff",
    "40282660": "239.tiff",
    "40282657": "225.tiff",
    "40282666": "226.tiff",
    "40282663": "241.tiff",
    "40282669": "230.tiff",
    "40282672": "228.tiff",
    "40282675": "232.tiff",
    "40282678": "243.tiff",
    "40282681": "234.tiff",
    "40282684": "236.tiff",
    "40282687": "238.tiff",
    "40282690": "249.tiff",
    "40282693": "247.tiff",
    "40282696": "250.tiff",
    "40282699": "252.tiff",
    "40282705": "254.tiff",
    "40282702": "256.tiff",
    "40282708": "240.tiff",
    "40282711": "258.tiff",
    "40282714": "242.tiff",
    "40282717": "244.tiff",
    "40282720": "245.tiff",
    "40282723": "246.tiff",
    "40282726": "262.tiff",
    "40282729": "264.tiff",
    "40282732": "251.tiff",
    "40282741": "266.tiff",
    "40282738": "270.tiff",
    "40282735": "268.tiff",
    "40282759": "272.tiff",
    "40282744": "276.tiff",
    "40282747": "274.tiff",
    "40282753": "248.tiff",
    "40282756": "253.tiff",
    "40282762": "255.tiff",
    "40282768": "259.tiff",
    "40282765": "257.tiff",
    "40282771": "260.tiff",
    "40282774": "261.tiff",
    "40282777": "278.tiff",
    "40282780": "280.tiff",
    "40282783": "288.tiff",
    "40282789": "286.tiff",
    "40282786": "284.tiff",
    "40282792": "290.tiff",
    "40282813": "263.tiff",
    "40282795": "265.tiff",
    "40282798": "267.tiff",
    "40282801": "282.tiff",
    "40282804": "292.tiff",
    "40282807": "269.tiff",
    "40282810": "273.tiff",
    "40282819": "271.tiff",
    "40282816": "294.tiff",
    "40282822": "275.tiff",
    "40282828": "277.tiff",
    "40282831": "296.tiff",
    "40282825": "298.tiff",
    "40282834": "300.tiff",
    "40282837": "304.tiff",
    "40282843": "302.tiff",
    "40282840": "281.tiff",
    "40282846": "306.tiff",
    "40282849": "279.tiff",
    "40282852": "285.tiff",
    "40282858": "283.tiff",
    "40282861": "308.tiff",
    "40282855": "289.tiff",
    "40282864": "287.tiff",
    "40282867": "291.tiff",
    "40282870": "314.tiff",
    "40282876": "293.tiff",
    "40282873": "312.tiff",
    "40282879": "310.tiff",
    "40282882": "316.tiff",
    "40282888": "295.tiff",
    "40282885": "317.tiff",
    "40282891": "319.tiff",
    "40282900": "299.tiff",
    "40282894": "297.tiff",
    "40282897": "303.tiff",
    "40282903": "301.tiff",
    "40282906": "322.tiff",
    "40282909": "326.tiff",
    "40282921": "324.tiff",
    "40282912": "327.tiff",
    "40282915": "305.tiff",
    "40282918": "307.tiff",
    "40282924": "330.tiff",
    "40282927": "332.tiff",
    "40282930": "335.tiff",
    "40282933": "309.tiff",
    "40282936": "337.tiff",
    "40282939": "311.tiff",
    "40282942": "339.tiff",
    "40282945": "315.tiff",
    "40282948": "318.tiff",
    "40282951": "313.tiff",
    "40282954": "342.tiff",
    "40282957": "320.tiff",
    "40282960": "341.tiff",
    "40282969": "323.tiff",
    "40282963": "321.tiff",
    "40282966": "344.tiff",
    "40282972": "346.tiff",
    "40282975": "348.tiff",
    "40282978": "325.tiff",
    "40282984": "350.tiff",
    "40282987": "352.tiff",
    "40282981": "354.tiff",
    "40282990": "329.tiff",
    "40282993": "328.tiff",
    "40283002": "356.tiff",
    "40282996": "333.tiff",
    "40282999": "331.tiff",
    "40283005": "334.tiff",
    "40283008": "338.tiff",
    "40283014": "336.tiff",
    "40283017": "357.tiff",
    "40283020": "361.tiff",
    "40283023": "359.tiff",
    "40283032": "363.tiff",
    "40283029": "340.tiff",
    "40283026": "365.tiff",
    "40283044": "347.tiff",
    "40283038": "343.tiff",
    "40283035": "372.tiff",
    "40283041": "351.tiff",
    "40283047": "349.tiff",
    "40283050": "374.tiff",
    "40283053": "376.tiff",
    "40283059": "367.tiff",
    "40283056": "353.tiff",
    "40283062": "355.tiff",
    "40283065": "380.tiff",
    "40283068": "378.tiff",
    "40283071": "345.tiff",
    "40283074": "358.tiff",
    "40283077": "385.tiff",
    "40283080": "389.tiff",
    "40283089": "387.tiff",
    "40283083": "362.tiff",
    "40283086": "360.tiff",
    "40283092": "366.tiff",
    "40283104": "397.tiff",
    "40283098": "391.tiff",
    "40283107": "393.tiff",
    "40283095": "368.tiff",
    "40283101": "364.tiff",
    "40283110": "395.tiff",
    "40283119": "369.tiff",
    "40283113": "371.tiff",
    "40283116": "370.tiff",
    "40283122": "399.tiff",
    "40283125": "402.tiff",
    "40283128": "404.tiff",
    "40283131": "373.tiff",
    "40283134": "375.tiff",
    "40283137": "377.tiff",
    "40283140": "410.tiff",
    "40283143": "408.tiff",
    "40283155": "413.tiff",
    "40283152": "415.tiff",
    "40283146": "379.tiff",
    "40283149": "381.tiff",
    "40283158": "382.tiff",
    "40283161": "417.tiff",
    "40283164": "406.tiff",
    "40283167": "383.tiff",
    "40283170": "419.tiff",
    "40283176": "421.tiff",
    "40283173": "388.tiff",
    "40283179": "384.tiff",
    "40283182": "390.tiff",
    "40283188": "386.tiff",
    "40283194": "423.tiff",
    "40283200": "392.tiff",
    "40283221": "429.tiff",
    "40283206": "431.tiff",
    "40283242": "427.tiff",
    "40283212": "425.tiff",
    "40283236": "394.tiff",
    "40283224": "432.tiff",
    "40283245": "396.tiff",
    "40283257": "433.tiff",
    "40283263": "435.tiff",
    "40283278": "398.tiff",
    "40283287": "400.tiff",
    "40283293": "401.tiff",
    "40283302": "403.tiff",
    "40283311": "405.tiff",
    "40283320": "436.tiff",
    "40283326": "407.tiff",
    "40283329": "438.tiff",
    "40283332": "440.tiff",
    "40283335": "409.tiff",
    "40283338": "411.tiff",
    "40283344": "442.tiff",
    "40283341": "444.tiff",
    "40283350": "412.tiff",
    "40283347": "446.tiff",
    "40283353": "450.tiff",
    "40283356": "448.tiff",
    "40283365": "414.tiff",
    "40283362": "416.tiff",
    "40283359": "452.tiff",
    "40283368": "418.tiff",
    "40283371": "420.tiff",
    "40283374": "454.tiff",
    "40283377": "456.tiff",
    "40283386": "422.tiff",
    "40283380": "426.tiff",
    "40283383": "457.tiff",
    "40283389": "424.tiff",
    "40283392": "428.tiff",
    "40283395": "461.tiff",
    "40283401": "459.tiff",
    "40283407": "430.tiff",
    "40283410": "464.tiff",
    "40283413": "466.tiff",
    "40283416": "468.tiff",
    "40283419": "434.tiff",
    "40283425": "439.tiff",
    "40283422": "470.tiff",
    "40283428": "437.tiff",
    "40283434": "441.tiff",
    "40283437": "472.tiff",
    "40283431": "445.tiff",
    "40283443": "474.tiff",
    "40283449": "476.tiff",
    "40283440": "443.tiff",
    "40283446": "447.tiff",
    "40283452": "449.tiff",
    "40283458": "478.tiff",
    "40283455": "482.tiff",
    "40283461": "451.tiff",
    "40283464": "480.tiff",
    "40283473": "455.tiff",
    "40283467": "486.tiff",
    "40283476": "484.tiff",
    "40283479": "453.tiff",
    "40283491": "492.tiff",
    "40283482": "491.tiff",
    "40283488": "458.tiff",
    "40283494": "489.tiff",
    "40283497": "460.tiff",
    "40283503": "463.tiff",
    "40283500": "494.tiff",
    "40283509": "462.tiff",
    "40283506": "465.tiff",
    "40283512": "497.tiff",
    "40283515": "501.tiff",
    "40283521": "467.tiff",
    "40283518": "499.tiff",
    "40283524": "469.tiff",
    "40283527": "471.tiff",
    "40283533": "503.tiff",
    "40283530": "505.tiff",
    "40283539": "473.tiff",
    "40283536": "475.tiff",
    "40283548": "507.tiff",
    "40283545": "477.tiff",
    "40283542": "509.tiff",
    "40283554": "511.tiff",
    "40283557": "514.tiff",
    "40283563": "479.tiff",
    "40283560": "481.tiff",
    "40283566": "515.tiff",
    "40283569": "517.tiff",
    "40283572": "483.tiff",
    "40283575": "487.tiff",
    "40283584": "520.tiff",
    "40283587": "488.tiff",
    "40283593": "485.tiff",
    "40283581": "522.tiff",
    "40283578": "524.tiff",
    "40283596": "490.tiff",
    "40283599": "493.tiff",
    "40283602": "526.tiff",
    "40283611": "528.tiff",
    "40283608": "530.tiff",
    "40283605": "495.tiff",
    "40283614": "496.tiff",
    "40283617": "498.tiff",
    "40283620": "532.tiff",
    "40283623": "535.tiff",
    "40283632": "500.tiff",
    "40283629": "538.tiff",
    "40283635": "540.tiff",
    "40283638": "542.tiff",
    "40283641": "502.tiff",
    "40283644": "504.tiff",
    "40283647": "506.tiff",
    "40283653": "544.tiff",
    "40283650": "508.tiff",
    "40283659": "548.tiff",
    "40283656": "510.tiff",
    "40283665": "546.tiff",
    "40283662": "550.tiff",
    "40283668": "512.tiff",
    "40283677": "513.tiff",
    "40283671": "552.tiff",
    "40283674": "516.tiff",
    "40283680": "518.tiff",
    "40283683": "521.tiff",
    "40283689": "519.tiff",
    "40283686": "523.tiff",
    "40283692": "525.tiff",
    "40283695": "527.tiff",
    "40283698": "531.tiff",
    "40283701": "529.tiff",
    "40283704": "533.tiff",
    "40283707": "534.tiff",
    "40283710": "537.tiff",
    "40283716": "541.tiff",
    "40283719": "536.tiff",
    "40283713": "539.tiff",
    "40283722": "543.tiff",
    "40283725": "545.tiff",
    "40283728": "547.tiff",
    "40283731": "549.tiff",
    "40283734": "551.tiff",
    "40283740": "553.tiff",
}
_MIDOG_JSON_FIGSHARE_ID = "41265615"

#: (domain, first slide index, last slide index + 1, subfolder) in the MIDOG++ annotation file
_MIDOG_DOMAINS = [
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


def _extract_midog_patches(
    image: Image.Image,
    img_data: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    """
    Save one 50x50 patch per annotation of a slide (label: its category), plus a background patch
    (label 0) shifted 100 px to the right if it overlaps no annotation.
    """
    img_id = img_data["id"]
    img_dims = (img_data["width"], img_data["height"])
    os.makedirs(out_dir, exist_ok=True)
    for ann in annotations:
        bbox = ann["bbox"]
        # boxes that are not exactly 50x50 px are ignored, as in the OpenMIBOOD scripts
        if bbox[2] - bbox[0] != 50 or bbox[3] - bbox[1] != 50:
            continue
        prefix = join(out_dir, f"{img_id:03d}_{ann['id']}")
        image.crop(bbox).save(f"{prefix}_{ann['category_id']}.tiff")

        background = [bbox[0] + 100, bbox[1], bbox[2] + 100, bbox[3]]
        if _overlaps_none(img_dims, annotations, background):
            image.crop(background).save(f"{prefix}_0.tiff")


def _download_midog(root: str) -> None:
    """
    Download the MIDOG++ slides one after another and cut them into patches. Each slide is
    deleted after its patches are extracted, and slides whose patches exist are skipped, so an
    interrupted download can be resumed.
    """
    tmp = join(root, "_tmp_midog")
    json_path = _download(
        f"https://ndownloader.figshare.com/files/{_MIDOG_JSON_FIGSHARE_ID}", tmp, "MIDOGpp.json"
    )
    with open(json_path) as f:
        midog = json.load(f)

    annotations = defaultdict(list)
    for ann in midog["annotations"]:
        annotations[ann["image_id"]].append(ann)
    fname_to_fid = {fname: fid for fid, fname in _MIDOG_FIGSHARE_FILES.items()}

    todo = []
    for domain, start, end, subfolder in _MIDOG_DOMAINS:
        for img_data in midog["images"][start:end]:
            out_dir = join(root, subfolder, domain, f"{img_data['id']:03d}")
            if not isdir(out_dir):
                todo.append((domain, out_dir, img_data))

    if todo:
        log.info(f"Downloading {len(todo)} MIDOG++ slides and extracting patches")

    for i, (domain, out_dir, img_data) in enumerate(todo):
        fname = img_data["file_name"]
        log.info(f"MIDOG++ slide {fname} (domain {domain}, {i + 1}/{len(todo)})")
        url = f"https://ndownloader.figshare.com/files/{fname_to_fid[fname]}"
        img_path = _download(url, tmp, fname)
        # patches are written to a temporary directory first, so that a slide whose extraction
        # was interrupted is processed again
        part_dir = out_dir + ".part"
        shutil.rmtree(part_dir, ignore_errors=True)
        with Image.open(img_path) as image:
            _extract_midog_patches(image, img_data, annotations[img_data["id"]], part_dir)
        os.replace(part_dir, out_dir)
        os.remove(img_path)

    shutil.rmtree(tmp, ignore_errors=True)


def _extract_ccagt_patches(coco: Dict[str, Any], images_dir: str, out_dir: str) -> None:
    """
    Save a 50x50 patch centered on each annotated nucleus that lies completely inside its image.
    Clusters and satellites (categories 2 and 3) lie inside nuclei and are skipped.
    """
    file_names = {img["id"]: img["file_name"] for img in coco["images"]}
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] not in (2, 3):
            by_image[ann["image_id"]].append(ann)

    for image_id, anns in by_image.items():
        file_name = file_names[image_id]
        slide_id = file_name.split("_")[0]
        src = join(images_dir, slide_id, f"{file_name}.jpg")
        if not exists(src):
            continue
        with Image.open(src) as img:
            for ann in anns:
                x, y, w, h = (int(c) for c in ann["bbox"])
                cx, cy = x + w // 2, y + h // 2
                box = [cx - 25, cy - 25, cx + 25, cy + 25]
                if box[0] < 0 or box[1] < 0 or box[2] >= img.width or box[3] >= img.height:
                    continue
                slide_dir = join(out_dir, file_name)
                os.makedirs(slide_dir, exist_ok=True)
                img.crop(box).save(join(slide_dir, f"{ann['id']}_{ann['category_id']}.jpg"))


def _download_ccagt(root: str) -> None:
    # Mendeley Data generates the archive on request, so it has no stable checksum
    archive = _download(
        "https://data.mendeley.com/public-api/zip/wg4bpm33hj/download/2", root, "ccagt.zip"
    )
    tmp = join(root, "_tmp_ccagt")
    shutil.rmtree(tmp, ignore_errors=True)
    _extract_zip(archive, tmp)
    images_dir = join(tmp, "subsets")
    for sub_zip in sorted(glob.glob(join(tmp, "wg4bpm33hj-2", "images", "*.zip"))):
        _extract_zip(sub_zip, images_dir)

    with open(join(tmp, "wg4bpm33hj-2", "CCAgT_COCO_OD.json")) as f:
        coco = json.load(f)

    dest = join(root, "far", "ccagt_crops")
    part_dir = dest + ".part"
    shutil.rmtree(part_dir, ignore_errors=True)
    _extract_ccagt_patches(coco, images_dir, part_dir)
    os.replace(part_dir, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    os.remove(archive)


MIDOG = _Source(
    "MIDOG++",
    tuple(join(subfolder, domain) for domain, _, _, subfolder in _MIDOG_DOMAINS),
    _download_midog,
)
CCAGT = _Source("CCAgT", ("far/ccagt_crops",), _download_ccagt)
FNAC2019 = _Source("FNAC 2019", ("far/fnac2019_crops",))


# =====================================================================
# OASIS-3 benchmark
# =====================================================================


def _download_chaos(root: str) -> None:
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise RuntimeError(
            "Converting CHAOS from DICOM to NIfTI requires 'SimpleITK' (pip install SimpleITK)."
        ) from e

    archive = _download(
        "https://zenodo.org/records/3431873/files/CHAOS_Test_Sets.zip?download=1",
        root,
        "CHAOS_Test_Sets.zip",
        md5="da6efc17118d7375654ab268473a5555",
    )
    tmp = join(root, "_tmp_chaos")
    shutil.rmtree(tmp, ignore_errors=True)
    _extract_zip(archive, tmp)

    def slice_number(path: str) -> int:
        # the DICOM files are named <prefix>-<slice number>.dcm
        return int(os.path.basename(path).split("-")[-1].split(".")[0])

    dest = join(root, "far", "CHAOS", "NIFTI", "InPhase")
    part_dir = dest + ".part"
    shutil.rmtree(part_dir, ignore_errors=True)
    os.makedirs(part_dir)
    mr_dir = join(tmp, "Test_Sets", "MR")
    for subject in sorted(os.listdir(mr_dir)):
        in_phase = join(mr_dir, subject, "T1DUAL", "DICOM_anon", "InPhase")
        files = sorted(glob.glob(join(in_phase, "*.dcm")), key=slice_number)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        sitk.WriteImage(reader.Execute(), join(part_dir, f"{subject}.nii.gz"))

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.replace(part_dir, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    os.remove(archive)


def _download_heart(root: str) -> None:
    # mirror of the Medical Segmentation Decathlon used by MONAI
    archive = _download(
        "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
        root,
        "Task02_Heart.tar",
        md5="06ee59366e1e5124267b774dbd654057",
    )
    _extract_tar(archive, join(root, "far", "SegmentationDecathlon"))
    os.remove(archive)


OASIS3 = _Source("OASIS-3", ("OASIS3",))
ATLAS = _Source("ATLAS R2.0", ("near/ATLAS",))
BRATS = _Source("BraTS 2023", ("near/BraTS2023-GLI",))
HEART = _Source(
    "Medical Segmentation Decathlon Task02 Heart",
    ("far/SegmentationDecathlon/Task02_Heart",),
    _download_heart,
)
CHAOS = _Source("CHAOS", ("far/CHAOS/NIFTI/InPhase",), _download_chaos)
