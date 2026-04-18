# src/ingestion/birads_reference.py

"""
BI-RADS Reference Knowledge Base

Sources: ACR BI-RADS Atlas (public lexicon descriptions),
         RadiologyInfo.org, Radiopaedia.org

This is publicly available educational/reference content,
not copyrighted clinical text.
"""

BIRADS_CATEGORIES = [
    {
        "source": "BI-RADS Atlas - Assessment Categories",
        "category": "BI-RADS 0",
        "finding_type": "assessment",
        "text": (
            "BI-RADS 0: Incomplete — Need Additional Imaging Evaluation and/or "
            "Prior Mammograms for Comparison. This category is used when the "
            "mammogram shows a finding that needs further evaluation before a "
            "final assessment can be made. Additional imaging may include spot "
            "compression views, magnification views, special mammographic views, "
            "or ultrasound. The radiologist may also request previous mammograms "
            "for comparison. This is common in screening mammography and does not "
            "necessarily indicate an abnormality — most callbacks result in benign "
            "findings after additional workup."
        )
    },
    {
        "source": "BI-RADS Atlas - Assessment Categories",
        "category": "BI-RADS 1",
        "finding_type": "assessment",
        "text": (
            "BI-RADS 1: Negative. The mammogram shows no significant findings. "
            "The breasts are symmetric with no masses, architectural distortion, "
            "or suspicious calcifications identified. Routine mammographic "
            "screening is recommended at the usual interval. This indicates "
            "a completely normal examination."
        )
    },
    {
        "source": "BI-RADS Atlas - Assessment Categories",
        "category": "BI-RADS 2",
        "finding_type": "assessment",
        "text": (
            "BI-RADS 2: Benign. A definite benign finding is identified. "
            "Examples include involuting calcified fibroadenomas, skin "
            "calcifications, metallic foreign bodies such as surgical clips, "
            "fat-containing lesions such as oil cysts, lipomas, galactoceles, "
            "and mixed-density hamartomas. Intramammary lymph nodes, vascular "
            "calcifications, implants, and architectural distortion clearly "
            "related to prior surgery are also benign findings. Routine screening "
            "is recommended."
        )
    },
    {
        "source": "BI-RADS Atlas - Assessment Categories",
        "category": "BI-RADS 3",
        "finding_type": "assessment",
        "text": (
            "BI-RADS 3: Probably Benign. A finding with a very high probability "
            "of being benign, with less than 2% likelihood of malignancy. "
            "Short-interval follow-up mammography at 6 months is recommended, "
            "followed by additional examinations at 12 and 24 months. Common "
            "findings include a non-palpable, circumscribed solid mass, a focal "
            "asymmetry, or a cluster of round or punctate calcifications. If the "
            "finding is stable over 2-3 years, it can be downgraded to BI-RADS 2."
        )
    },
    {
        "source": "BI-RADS Atlas - Assessment Categories",
        "category": "BI-RADS 4",
        "finding_type": "assessment",
        "text": (
            "BI-RADS 4: Suspicious. A finding that does not have the classic "
            "appearance of malignancy but is sufficiently suspicious to warrant "
            "tissue diagnosis (biopsy). The likelihood of malignancy ranges from "
            "2% to 95%. This category is often subdivided: 4A (low suspicion, "
            "2-10%), 4B (moderate suspicion, 10-50%), and 4C (high suspicion, "
            "50-95%). Tissue diagnosis is recommended for all subcategories."
        )
    },
    {
        "source": "BI-RADS Atlas - Assessment Categories",
        "category": "BI-RADS 5",
        "finding_type": "assessment",
        "text": (
            "BI-RADS 5: Highly Suggestive of Malignancy. The finding has a "
            "95% or higher likelihood of being malignant. Appropriate action "
            "should be taken, typically tissue diagnosis and treatment. "
            "Classic findings include spiculated irregular masses, fine linear "
            "or fine-linear-branching calcifications in a segmental distribution, "
            "and irregularly shaped masses with associated pleomorphic "
            "calcifications."
        )
    },
    {
        "source": "BI-RADS Atlas - Assessment Categories",
        "category": "BI-RADS 6",
        "finding_type": "assessment",
        "text": (
            "BI-RADS 6: Known Biopsy-Proven Malignancy. This category is "
            "reserved for findings on imaging performed after biopsy has proven "
            "malignancy but before definitive therapy such as surgical excision, "
            "radiation, or chemotherapy. It is used to evaluate response to "
            "neoadjuvant chemotherapy or to confirm the location of the known "
            "malignancy prior to surgical excision."
        )
    },
]


CALCIFICATION_DESCRIPTORS = [
    {
        "source": "BI-RADS Atlas - Calcification Morphology",
        "category": "Typically Benign Calcifications",
        "finding_type": "calcification_morphology",
        "text": (
            "Typically benign calcification morphologies include: "
            "Skin calcifications — typically lucent-centered, polygonal deposits "
            "located along the skin surface. "
            "Vascular calcifications — linear, parallel track or tubular calcifications "
            "associated with blood vessel walls. "
            "Coarse or popcorn-like calcifications — large, dense, lobular calcifications "
            "typically representing involuting fibroadenomas. "
            "Large rod-like calcifications — solid or occasionally lucent-centered, "
            "smooth linear deposits, often oriented toward the nipple, representing "
            "secretory disease or plasma cell mastitis. "
            "Round calcifications — well-defined, spherical calcifications that may vary "
            "in size; when small (less than 0.5mm), they are called punctate. "
            "Rim calcifications — thin, curvilinear deposits on the surface of a sphere, "
            "typically seen in fat necrosis or calcified cysts. "
            "Dystrophic calcifications — irregular, typically greater than 0.5mm, "
            "commonly forming in irradiated or traumatized breast tissue. "
            "Milk of calcium — sedimented calcifications in cysts, "
            "best seen on lateral views as crescent-shaped deposits."
        )
    },
    {
        "source": "BI-RADS Atlas - Calcification Morphology",
        "category": "Suspicious Calcifications",
        "finding_type": "calcification_morphology",
        "text": (
            "Suspicious calcification morphologies requiring further evaluation: "
            "Amorphous calcifications — sufficiently small or hazy that a more specific "
            "particle shape cannot be determined. They can be benign or malignant. "
            "When grouped and bilateral, they are more likely benign; when unilateral "
            "or in a segmental distribution, suspicion increases. "
            "Coarse heterogeneous calcifications — irregular, conspicuous calcifications "
            "that are generally between 0.5mm and 1.0mm and do not conform to benign shapes. "
            "They may be forming fibroadenoma, post-traumatic, or associated with DCIS. "
            "Fine pleomorphic calcifications — usually less than 0.5mm, varying in size "
            "and shape, without specific benign characteristics. More concerning than "
            "amorphous forms. "
            "Fine linear or fine-linear branching calcifications — thin, irregular, "
            "discontinuous linear deposits, sometimes branching, less than 0.5mm in width. "
            "Their appearance suggests filling of the lumen of a duct irregularly "
            "involved by breast cancer (DCIS). These carry the highest suspicion "
            "for malignancy."
        )
    },
    {
        "source": "BI-RADS Atlas - Calcification Distribution",
        "category": "Calcification Distribution Patterns",
        "finding_type": "calcification_distribution",
        "text": (
            "Calcification distribution describes how calcifications are arranged: "
            "Diffuse — scattered randomly throughout the breast. Usually benign, "
            "representing bilateral processes. "
            "Regional — occupying a large volume of tissue (more than 2cc), not "
            "necessarily conforming to a duct distribution. May represent a broader "
            "process and is sometimes suspicious. "
            "Grouped or clustered — at least 5 calcifications within 1cc of tissue. "
            "The most common distribution for suspicious findings. "
            "Linear — arranged in a line, suggesting deposition in a duct. Increases "
            "suspicion, especially with suspicious morphology. "
            "Segmental — deposited in calcifications within a duct and its branches, "
            "suggesting an intraductal process. Raises concern for extensive DCIS "
            "involving a single lobe or segment of the breast."
        )
    },
]


MASS_DESCRIPTORS = [
    {
        "source": "BI-RADS Atlas - Mass Shape",
        "category": "Mass Shape Descriptors",
        "finding_type": "mass_shape",
        "text": (
            "Mass shape describes the overall form of a lesion: "
            "Oval — elliptical or egg-shaped, may include 2-3 undulations. "
            "Generally associated with benign lesions such as fibroadenomas or cysts. "
            "Round — spherical, ball-shaped, circular. Commonly seen with cysts, "
            "fibroadenomas, and some well-circumscribed carcinomas. "
            "Irregular — neither round nor oval; the shape is uneven and cannot be "
            "characterized by the other descriptors. This shape is more often "
            "associated with malignancy and typically warrants biopsy."
        )
    },
    {
        "source": "BI-RADS Atlas - Mass Margins",
        "category": "Mass Margin Descriptors",
        "finding_type": "mass_margins",
        "text": (
            "Mass margins describe the edge characteristics of a lesion: "
            "Circumscribed — margins are well-defined and sharply demarcated, with "
            "an abrupt transition between the lesion and surrounding tissue. More than "
            "75% of the margin must be well-defined. Associated with benign findings. "
            "Obscured — margins are hidden by adjacent or superimposed tissue. Cannot "
            "be adequately assessed. "
            "Microlobulated — margins have short-cycle undulations creating a scalloped "
            "border. Raises suspicion for malignancy. "
            "Indistinct — margins are poorly defined, suggesting infiltration into "
            "surrounding tissue. Suspicious for malignancy. "
            "Spiculated — lines radiating from the margin of the mass, creating a "
            "starburst or sunburst pattern. Highly suspicious for malignancy with "
            "strong positive predictive value."
        )
    },
    {
        "source": "BI-RADS Atlas - Mass Density",
        "category": "Mass Density Descriptors",
        "finding_type": "mass_density",
        "text": (
            "Mass density compares the x-ray attenuation of the lesion to an equal "
            "volume of fibroglandular tissue: "
            "High density — the lesion is denser than surrounding tissue. While not "
            "specific, high density masses that are also irregular or spiculated "
            "raise concern. "
            "Equal density (isodense) — the lesion has the same density as surrounding "
            "fibroglandular tissue. This is the most common presentation. "
            "Low density — the lesion is less dense than surrounding tissue but not "
            "fat-containing. "
            "Fat-containing — includes radiolucent areas representing fat within the "
            "lesion. Strongly suggests benignity — oil cysts, lipomas, galactoceles, "
            "and hamartomas are fat-containing."
        )
    },
]


ASSOCIATED_FEATURES = [
    {
        "source": "BI-RADS Atlas - Associated Features",
        "category": "Associated Features",
        "finding_type": "associated_features",
        "text": (
            "Associated features are additional findings that may accompany masses "
            "or calcifications: "
            "Skin retraction — pulling inward of the skin surface, caused by a lesion "
            "tethering the overlying skin. Can be a sign of malignancy. "
            "Nipple retraction — pulling inward of the nipple, may be new or longstanding. "
            "New nipple retraction raises suspicion. "
            "Skin thickening — may be focal or diffuse. Diffuse thickening can indicate "
            "inflammatory breast cancer, radiation changes, or systemic conditions. "
            "Trabecular thickening — thickening of the Cooper ligaments, which may "
            "indicate edema from inflammatory cancer or lymphatic obstruction. "
            "Axillary lymphadenopathy — enlarged or abnormal axillary lymph nodes. "
            "While often reactive, can indicate metastatic disease. "
            "Architectural distortion — distortion of the normal breast parenchyma "
            "with no definite visible mass. May be caused by prior surgery, biopsy, "
            "or an underlying malignancy such as invasive lobular carcinoma."
        )
    },
]


def get_all_birads_documents():
    """Return all BI-RADS reference documents as a flat list."""
    all_docs = (
        BIRADS_CATEGORIES +
        CALCIFICATION_DESCRIPTORS +
        MASS_DESCRIPTORS +
        ASSOCIATED_FEATURES
    )
    print(f"Total BI-RADS reference documents: {len(all_docs)}")
    return all_docs


def save_birads_reference(output_path="data/birads_reference/birads_knowledge.json"):
    """Save BI-RADS reference to JSON."""
    import os, json
    docs = get_all_birads_documents()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)
    print(f"💾 Saved BI-RADS reference to {output_path}")
    return docs


if __name__ == "__main__":
    docs = save_birads_reference()
    print(f"\nCategories covered:")
    for doc in docs:
        print(f"  - {doc['category']}")