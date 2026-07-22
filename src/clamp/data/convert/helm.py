"""HELM-notation-to-SMILES fallback.

NOT YET IMPLEMENTED BY DESIGN. MD_design_docs/06_task5_data_pipeline_plan.md
§2.2/§7.3 flags the HELM toolchain choice as unverified — candidates
(Pistoia Alliance's open-source HELM toolkit, RDKit's own limited HELM
parsing) need a short hands-on spike against the modification vocabulary
actually present in the pulled data before picking one. Doc 09 §6/§16
explicitly says not to wire a library in ahead of that spike.
"""


def convert_via_helm(helm_string: str) -> str:
    raise NotImplementedError(
        "HELM toolchain not yet selected — see MD_design_docs/06_task5_data_pipeline_plan.md §7.3"
    )
