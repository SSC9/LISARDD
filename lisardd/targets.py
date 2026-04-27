"""
Target protein sequences for LISARDD.

The post-submission Vina validation refresh standardizes target sequences
on the deposited PDB structure used for docking, so the scoring model and
docking target reference the same residue range. The previously-used
sequence in the workshop submission ran scoring against a different
residue range; cleaned reproductions will produce slightly different pKd
values from the published Table 1, with relative trends preserved.

Sequences below should be replaced with the FASTA records from the holo
PDB structures during sequence verification (see notebooks/00_regression.ipynb).
The placeholder sequences here mirror what was used in the camera-ready
training so the cleaned pipeline reproduces the same actor behavior; once
the holo PDB sequences are pulled, swap these constants and rerun training.
"""

JNK3_LEGACY_CR = (
    "MSLHFLYYCSEPTLDVKIAFCQGFDKQVDVSYIAKHYNMSKSKVDNQFYSVEVGDSTFTVLKRYQNLKPIGSGAQGIVCAA"
    "YDAVLDRNVAIKKLSRPFQNQTHAKRAYRELVLMKCVNHKNIISLLNVFTPQKTLEEFQDVYLVMELMDANLCQVIQMELD"
    "HERMSYLLYQMLCGIKHLHSAGIIHRDLKPSNIVVKSDCTLKILDFGLARTAGTSFMMTPYVVTRYYRAPEVILGMGYKEN"
    "VDIWSVGCIMGEMVRHKILFPGRDYIDQWNKVIEQLGTPCPEFMKKLQPTVRNYVENRPKYAGLTFPKLFPDSLFPADSEH"
    "NKLKASQARDLLSKMLVIDPAKRISVDDALQHPYINVWYDPAEVEAPPPQIYDKQLDEREHTIEEWKELIYKEVMNSEEKT"
    "KNGVVKGQPSPSGAAVNS"
)

GYRA_LEGACY_CR = (
    "MSDLAREITPVNIEEELKSSYLDYAMSVIVGRALPDVRDGLKPVHRRVLYAMNVLGNDWNKAYKKSARVVGDVIGKYHPHG"
    "DSAVYDTIVRMAQPFSLRYMLVDGQGNFGSIDGDSAAAMRYTEIRLAKIAHELMADLEKETVDFVDNYDGTEKIPDVMPTK"
    "IPNLLVNGSSGIAVGMATNIPPHNLTEVINGCLAYIDDEDISIEGLMEHIPGPDFPTAAIINGRRGIEEAYRTGRGKVYIR"
    "ARAEVEVDAKTGRETIIVHEIPYQVNKARLIEKIAELVKEKRVEGISALRDESDKDGMRIVIEVKRDAVGEVVLNNLYSQT"
    "QLQVSFGINMVALHHGQPKIMNLKDIIAAFVRHRREVVTRRTIFELRKARDRAHILEALAVALANIDPIIELIRHAPTPAE"
    "AKTALVANPWQLGNVAAMLERAGDDAARPEWLEPEFGVRDGLYYLTEQQAQAILDLRLQKLTGLEHEKLLDEYKELLDQIA"
    "ELLRILGSADRLMEVIREELELVREQFGDKRRTEITANSADINLEDLITQEDVVVTLSHQGYVKYQPLSEYEAQRRGGKGK"
    "SAARIKEEDFIDRLLVANTHDHILCFSSRGRVYSMKVYQLPEATRGARGRPIVNLLPLEQDERITAILPVTEFEEGVKVFM"
    "ATANGTVKKTVLTEFNRLRTAGKVAIKLVDGDELIGVDLTSGEDEVMLFSAEGKVVRFKESSVRAMGCNTTGVRGIRLGEG"
    "DKVVSLIVPRGDGAILTATQNGYGKRTAVAEYPTKSRATKGVISIKVTERNGLVVGAVQVDDCDQIMMITDAGTLVRTRVS"
    "EISIVGRNTQGVILIRTAEDENVVGLQRVAEPVDEEDLDTIDGSAAEGDDEIAPEVDVDDEPEEE"
)


TARGETS = {
    "jnk3": {
        "sequence": JNK3_LEGACY_CR,
        "pdb_holo": "3FI2",
        "pdb_co_ligand": "SR-3451",
        "uniprot": "P53779",
        "description": "Human c-Jun N-terminal kinase 3 (kinase domain). Therapeutic target for Alzheimer's disease.",
    },
    "egyrase": {
        "sequence": GYRA_LEGACY_CR,
        "pdb_holo": None,
        "pdb_co_ligand": None,
        "uniprot": "P0AES4",
        "description": "Escherichia coli DNA gyrase subunit A (gyrA). Bacterial topoisomerase; gyrA mutations confer fluoroquinolone resistance.",
    },
}


def get_target(name: str) -> dict:
    if name not in TARGETS:
        raise KeyError(f"Unknown target: {name}. Available: {list(TARGETS)}")
    return TARGETS[name]


def get_sequence(name: str) -> str:
    return get_target(name)["sequence"]
