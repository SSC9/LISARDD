"""
Target protein sequences for LISARDD.

The post-submission Vina validation refresh standardizes target sequences
on the deposited holo PDB structures used for docking, so the scoring
model and docking target reference the same residue range. New runs
produce pKd values calibrated against this aligned reference; trajectories
will differ slightly from the camera-ready Table 1 values, with relative
trends preserved.

Active targets (used by the cleaned pipeline):
    jnk3   : PDB 3FI2 chain A, 353 aa kinase catalytic domain (6 'X' residues
             mark a disordered loop). Holo with bound SR-3451 (PDB resname
             SR2) at the ATP site. UniProt P53779.

    egyrase: PDB 1AB4 chain A, 493 aa N-terminal 59 kDa fragment of
             E. coli DNA gyrase A. Apo. Contains the quinolone-binding
             pocket (residues around Ser83 / Asp87). UniProt P0AES4.
             Vina box derivation for this apo target is handled in the
             validation pipeline (Stage 6) using literature pocket
             coordinates or homology to S. aureus quinolone-bound holos
             (e.g., PDB 2XCS / 5CDQ).

Legacy sequences from the camera-ready submission are retained as
*_LEGACY_CR constants for reproducibility and for the regression notebook,
which loads camera-ready actor weights and verifies architecture
equivalence against archived pickles in
ICML_2025_Workshop_Submission_Artifacts/pickles/.
"""

JNK3_3FI2 = (
    "MSKSKVDNQFYSVEVGDSTFTVLKRYQNLKPIGSGAQGIVCAAYDAVLDRNVAIKKLSRPFQNQTHAKRAYRELVLMKCV"
    "NHKNIISLLNVFTPQKTLEEFQDVYLVMELMDANLCQVIQMELDHERMSYLLYQMLCGIKHLHSAGIIHRDLKPSNIVVK"
    "SDCTLKILDFGLARTAGTSFMMTPYVVTRYYRAPEVILGMGYKENVDIWSVGCIMGEMVRHKILFPGRDYIDQWNKVIEQ"
    "LGTPCPEFMKKLQPTVRNYVENRPKYAGLTFPKLFPDSLFPADSEHNKLKASQARDLLSKMLVIDPAKRISVDDALQHPY"
    "INVWYXXXXXXDEREHTIEEWKELIYKEVMNSE"
)

GYRA_1AB4 = (
    "VGRALPDVRDGLKPVHRRVLYAMNVLGNDWNKAYKKSARVVGDVIGKYHPHGDSAVYDTIVRMAQPFSLRYMLVDGQGNF"
    "GSIDGDSAAAMRYTEIRLAKIAHELMADLEKETVDFVDNYDGTEKIPDVMPTKIPNLLVNGSSGIAVGMATNIPPHNLTE"
    "VINGCLAYIDDEDISIEGLMEHIPGPDFPTAAIINGRRGIEEAYRTGRGKVYIRARAEVEVDAKTGRETIIVHEIPYQVN"
    "KARLIEKIAELVKEKRVEGISALRDESDKDGMRIVIEVKRDAVGEVVLNNLYSQTQLQVSFGINMVALHHGQPKIMNLKD"
    "IIAAFVRHRREVVTRRTIFELRKARDRAHILEALAVALANIDPIIELIRHAPTPAEAKTALVANPWQLGNVAAMLERAGD"
    "DAARPEWLEPEFGVRDGLYYLTEQQAQAILDLRLQKLTGLEHEKLLDEYKELLDQIAELLRILGSADRLMEVIREELELV"
    "REQFGDKRRTEIT"
)

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
        "sequence": JNK3_3FI2,
        "legacy_sequence": JNK3_LEGACY_CR,
        "length": len(JNK3_3FI2),
        "pdb_holo": "3FI2",
        "pdb_chain": "A",
        "ligand_resname": "SR2",
        "ligand_name": "SR-3451 (aminopyrazole inhibitor)",
        "binding_site": "ATP site",
        "uniprot": "P53779",
        "description": "Human c-Jun N-terminal kinase 3 (kinase domain). Therapeutic target for Alzheimer's disease. Sequence is the deposited 3FI2 chain A FASTA; six 'X' residues mark a disordered loop not resolved in the crystal.",
    },
    "egyrase": {
        "sequence": GYRA_1AB4,
        "legacy_sequence": GYRA_LEGACY_CR,
        "length": len(GYRA_1AB4),
        "pdb_holo": "1AB4",
        "pdb_chain": "A",
        "ligand_resname": None,
        "ligand_name": None,
        "binding_site": "quinolone pocket near Ser83 / Asp87",
        "uniprot": "P0AES4",
        "description": "E. coli DNA gyrase subunit A. Sequence is the deposited 1AB4 chain A FASTA, a 59 kDa N-terminal fragment containing the quinolone-binding pocket. 1AB4 is apo; box derivation for the validation pipeline uses literature coordinates or homology mapping from a quinolone-bound holo (e.g., S. aureus 2XCS / 5CDQ).",
    },
}


def get_target(name: str) -> dict:
    if name not in TARGETS:
        raise KeyError(f"Unknown target: {name}. Available: {list(TARGETS)}")
    return TARGETS[name]


def get_sequence(name: str) -> str:
    return get_target(name)["sequence"]


def get_legacy_sequence(name: str) -> str:
    """Return the camera-ready sequence (for the regression notebook)."""
    return get_target(name)["legacy_sequence"]
